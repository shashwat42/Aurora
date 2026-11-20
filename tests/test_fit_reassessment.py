"""Analytic and independent least-squares checks for diagnostic fit bounds."""
import importlib.util
from pathlib import Path
import unittest
import json
from tempfile import TemporaryDirectory
import contextlib
import io

import numpy as np

spec = importlib.util.spec_from_file_location(
    'reassess', Path(__file__).resolve().parents[1] / 'scripts/reassess_rgb_fit.py')
reassess = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reassess)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_global_affine_pilot as pilot
from aurora.manifest import file_hash
from PIL import Image


class FitBounds(unittest.TestCase):
    def test_ridge_satisfies_normal_equations_with_constant_feature(self):
        rng = np.random.default_rng(12)
        x, y = rng.normal(size=(30, 15)), rng.normal(size=(30, 6))
        x[:, 4] = 0.3
        model = pilot.fit_ridge(x, y, 1.0)
        z = (x - model['center']) / model['scale']
        weights = np.asarray(model['weights'])
        residual = z @ weights + model['intercept'] - y
        np.testing.assert_allclose(z.T @ residual + weights, 0, atol=1e-12)
        np.testing.assert_allclose(residual.mean(0), 0, atol=1e-12)
        self.assertTrue(np.isfinite(weights).all())

    def test_tiny_pilot_ignores_holdouts_and_diagnostic_targets_when_predicting(self):
        root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory(dir=root / 'artifacts/tests') as temp:
            directory = Path(temp)
            rows = []
            rng = np.random.default_rng(29)
            for i, split in enumerate(['train', 'train', 'diagnostic', 'selection', 'test']):
                row = {'id': str(i), 'scene_id': str(i), 'split': split}
                for key in ['raw', 'reference']:
                    path = directory / f'{i}-{key}.png'
                    if split in ['selection', 'test']:
                        # Deliberately absent: any accidental holdout read fails.
                        digest = 'unused'
                    else:
                        Image.fromarray(rng.integers(20, 210, (13, 17, 3), dtype=np.uint8)).save(path)
                        digest = file_hash(path)
                    row[key + '_path'], row[key + '_sha256'] = path.name, digest
                rows.append(row)
            manifest = directory / 'manifest.json'
            def write_manifest():
                manifest.write_text(json.dumps({'status': 'frozen', 'pairs': rows, 'challenging': []}))
                manifest.with_suffix('.json.sha256').write_text(file_hash(manifest))
            write_manifest()
            control = directory / 'control.json'
            control.write_text(json.dumps({'records': [{'id': '2', 'psnr': 10., 'ssim': .5}]}))
            cfg = json.loads((root / 'configs/global_affine_pilot.json').read_text()) | {
                'manifest': str(manifest), 'data_root': str(directory),
                'expected_train_count': 2, 'expected_diagnostic_count': 1,
                'control_summary': str(control), 'control_summary_sha256': file_hash(control), 'threads': 1}
            predictions, models = [], []
            for index in range(2):
                out = directory / f'run-{index}'
                config = directory / f'config-{index}.json'
                config.write_text(json.dumps(cfg | {'manifest_sha256': file_hash(manifest), 'output': str(out)}))
                with contextlib.redirect_stdout(io.StringIO()):
                    result = pilot.main(config)
                self.assertFalse(result['selected'])
                self.assertEqual(result['diagnostic_count'], 1)
                model = json.loads((out / 'model.json').read_text())
                self.assertEqual(model['train_ids'], ['0', '1'])
                models.append({k: model[k] for k in ['weights', 'center', 'scale', 'intercept']})
                predictions.append((out / 'images/2.png').read_bytes())
                with self.assertRaises(FileExistsError):
                    pilot.main(config)
                # Change diagnostic target only; fitted parameters and output must persist.
                ref = directory / rows[2]['reference_path']
                Image.new('RGB', (17, 13), (200, 170, 140)).save(ref)
                rows[2]['reference_sha256'] = file_hash(ref)
                write_manifest()
            self.assertEqual(models[0], models[1])
            self.assertEqual(predictions[0], predictions[1])

    def test_affine_matches_independent_lstsq_and_constant_channel(self):
        x = np.random.default_rng(17).random((9, 13, 3))
        x[..., 2] = 0.4
        y = np.clip(x * [0.5, 0.7, 0.9] + [0.1, 0.2, 0.05], 0, 1)
        result, _, _ = reassess.affine_oracle(x, y)
        for c in range(3):
            design = np.column_stack([x[..., c].ravel(), np.ones(117)])
            expected = (design @ np.linalg.lstsq(design, y[..., c].ravel(), rcond=None)[0]).reshape(9, 13)
            np.testing.assert_allclose(result[..., c], expected, atol=1e-12)

    def test_cap_bound_matches_closest_feasible_projection(self):
        x = np.array([[[0.0, 0.8, 0.4], [1.0, 0.1, 0.6]]])
        y = 1 - x
        closest = np.clip(y, np.maximum(0, x - 0.5), np.minimum(1, x + 0.5))
        bound = reassess.residual_floor(x, y)
        self.assertAlmostEqual(bound['mse_infimum'], float(np.mean((closest - y) ** 2)))
        self.assertAlmostEqual(bound['unreachable_channel_fraction'], 4 / 6)

    def test_uniform_bias_has_no_centered_error(self):
        y = np.zeros((5, 7, 3))
        result = reassess.components(y + [0.1, -0.2, 0.3], y)
        self.assertAlmostEqual(result['mse'], 0.14 / 3)
        self.assertAlmostEqual(result['mean_bias_fraction'], 1)
        self.assertAlmostEqual(result['centered_mse'], 0)
