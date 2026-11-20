import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image
import torch

from aurora.dataset import PairedDataset
from aurora.evaluation.rgb import evaluate_bundle
from aurora.inference import RGBInference
from aurora.manifest import file_hash
from aurora.models import SceneAffineUNet
from aurora.models.scene_rgb import scene_statistics
from aurora.training.run import train, validate_config

ROOT = Path(__file__).resolve().parents[1]


class SceneRGB(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_identity_native_geometry_moments_and_invalid_context(self):
        model = SceneAffineUNet(base_channels=4)
        for shape in [(1, 1), (17, 31)]:
            x = torch.rand(2, 3, *shape)
            self.assertTrue(torch.equal(model(x), x))
            stats = scene_statistics(x)
            self.assertTrue(torch.equal(model(x), model(x, scene_stats=stats)))
            reference = x.numpy().astype(np.float64)
            expected = np.concatenate([reference.mean((2, 3)), reference.std((2, 3))], axis=1)
            np.testing.assert_allclose(stats.detach().numpy(), expected, atol=1e-7)
            torch.testing.assert_close(stats, scene_statistics(x.flip(-1).rot90(1, (-2, -1))))
        for invalid in [torch.zeros(2, 5), torch.full((2, 6), float('nan')), torch.full((2, 6), -1.)]:
            with self.assertRaises(ValueError):
                model(x, scene_stats=invalid)

    def test_same_scene_colour_mapping_on_crop_and_native_and_branch_gradients(self):
        torch.manual_seed(0)
        model = SceneAffineUNet(base_channels=4)
        x = torch.rand(1, 3, 48, 56) * .3 + .2
        stats = scene_statistics(x)
        with torch.no_grad():
            model.scene[-1].bias.fill_(.2)
            model.scene[-1].weight.fill_(.03)
        native = model.forward_unclamped(x)
        crop = model.forward_unclamped(x[..., 8:40, 8:40], scene_stats=stats)
        torch.testing.assert_close(native[..., 8:40, 8:40], crop, atol=0, rtol=0)
        # With zero local head the mapping is spatially affine; no context seam.
        self.assertGreater((native-x).abs().mean().item(), 0)
        model.zero_grad(set_to_none=True)
        (native - .1).abs().mean().backward()
        self.assertGreater(model.scene[-1].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.local.head.weight.grad.abs().sum().item(), 0)
        with torch.no_grad():
            model.scene[-1].bias.fill_(100)
            model.local.head.bias.fill_(100)
        self.assertTrue((model(torch.ones_like(x)) == 1).all())

    def test_scene_configuration_rejects_wrong_context_and_warm_start(self):
        config = json.loads((ROOT / 'configs/rgb_baseline_v3.json').read_text())
        validate_config(config)
        for changed in [{'scene_context': None}, {'init_bundle': 'anything'},
                        {'identity_weight': .5}, {'loss_margin': 64}, {'l1_prediction': 'bounded'}]:
            with self.assertRaises(ValueError):
                validate_config(config | changed)

    def test_tiny_training_full_scene_stats_reload_diagnostics_and_selection_guard(self):
        (ROOT / 'artifacts/tests').mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=ROOT / 'artifacts/tests') as temp:
            root = Path(temp)
            rng = np.random.default_rng(21)
            rows = []
            for index, split in enumerate(['train', 'diagnostic', 'selection', 'test']):
                row = {'id': str(index), 'scene_id': str(index), 'split': split}
                raw = rng.integers(20, 180, (33, 41, 3), dtype=np.uint8)
                for name, pixels in [('raw', raw), ('reference', raw + 20)]:
                    path = root / f'{index}-{name}.png'
                    Image.fromarray(pixels).save(path)
                    row[name + '_path'], row[name + '_sha256'] = path.name, file_hash(path)
                rows.append(row)
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'status': 'frozen', 'pairs': rows, 'challenging': []}))
            manifest.with_suffix('.json.sha256').write_text(file_hash(manifest))
            data = PairedDataset(root, manifest, crop_size=16, include_scene_stats=True)
            raw = np.asarray(Image.open(root / '0-raw.png'), dtype=np.float32) / 255
            expected = scene_statistics(torch.from_numpy(raw.transpose(2, 0, 1).copy()).unsqueeze(0))[0]
            for epoch in [0, 1]:
                data.set_epoch(epoch)
                batch = data[0]
                self.assertTrue(torch.equal(batch['scene_stats'], expected))
                self.assertFalse(torch.allclose(batch['scene_stats'], scene_statistics(batch['input'].unsqueeze(0))[0]))
            config = json.loads((ROOT / 'configs/rgb_baseline_v3.json').read_text()) | {
                'manifest': str(manifest), 'manifest_sha256': file_hash(manifest), 'data_root': str(root),
                'model': {'base_channels': 4, 'residual_scale': .5, 'scene_hidden': 8, 'gain_scale': .5, 'offset_scale': .25},
                'crop_size': 16, 'batch_size': 1, 'epochs': 2, 'candidate_epochs': [1, 2],
                'threads': 2, 'output': str(root / 'run'), 'diagnostic_gate_record': str(root / 'gate.json')}
            path = root / 'config.json'; path.write_text(json.dumps(config))
            with contextlib.redirect_stdout(io.StringIO()):
                result = train(path)
            self.assertEqual(result['steps'], 2)
            self.assertTrue(result['finite_losses'] and result['checkpoint_reload_exact'])
            bundle = root / 'run/last/bundle.json'
            inference = RGBInference(bundle)
            self.assertIsInstance(inference.model, SceneAffineUNet)
            with contextlib.redirect_stdout(io.StringIO()):
                scored = evaluate_bundle(bundle, 'diagnostic', root / 'diagnostic', threads=2)
            self.assertEqual(scored['count'], 1)
            self.assertEqual((scored['records'][0]['width'], scored['records'][0]['height']), (41, 33))
            with self.assertRaisesRegex(ValueError, 'Diagnostic acceptance'):
                evaluate_bundle(bundle, 'selection', root / 'selection')
            self.assertFalse((root / 'selection').exists())
            (root / 'gate.json').write_text(json.dumps({
                'status': 'diagnostic_accepted_for_selection', 'config_sha256': 'wrong',
                'manifest_sha256': config['manifest_sha256'], 'fixed_native_review_passed': True,
                'candidate_weights_sha256': [inference.bundle['weights_sha256']]}))
            with self.assertRaisesRegex(ValueError, 'Diagnostic acceptance'):
                evaluate_bundle(bundle, 'selection', root / 'selection')
            changed_settings = json.loads(bundle.read_text())
            for settings in [changed_settings['model'], changed_settings['configuration']['model']]:
                settings['offset_scale'] = .2
            variant = bundle.parent / 'variant.json'; variant.write_text(json.dumps(changed_settings))
            with self.assertRaisesRegex(ValueError, 'different weights/data/metrics'):
                evaluate_bundle(variant, 'diagnostic', root / 'diagnostic', threads=2)
            for split in ['test', 'challenging']:
                with self.assertRaises(ValueError):
                    evaluate_bundle(bundle, split, root / split)
            changed = json.loads(bundle.read_text())
            changed['configuration']['architecture'] = 'rgb-residual-unet-v1'
            bad = bundle.parent / 'bad.json'; bad.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'differs from configuration'):
                RGBInference(bad)
