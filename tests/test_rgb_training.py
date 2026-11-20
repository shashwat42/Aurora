from copy import deepcopy
from io import BytesIO
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image, ImageCms
from scipy.signal import convolve2d
import torch

from aurora.image import validate_image
from aurora.inference import RGBInference, export_png
from aurora.models import ResidualUNet
from aurora.training.metrics import psnr, ssim
from aurora.training.run import train, validate_config

ROOT = Path(__file__).resolve().parents[1]


class RGBTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        (ROOT / "artifacts/tests").mkdir(parents=True, exist_ok=True)
        self.temp = TemporaryDirectory(dir=ROOT / "artifacts/tests")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_native_geometry_identity_initialization_and_range(self):
        model = ResidualUNet(base_channels=4).eval()
        with torch.inference_mode():
            for h, w in [(1, 1), (7, 19), (31, 17), (16, 16), (17, 31)]:
                x = torch.rand(1, 3, h, w)
                self.assertTrue(torch.equal(model(x), x))
            model.head.bias.fill_(100)
            y = model(torch.rand(1, 3, 17, 31))
            self.assertTrue(torch.isfinite(y).all())
            self.assertTrue(((y >= 0) & (y <= 1)).all())

    def test_model_rejects_invalid_pixels(self):
        model = ResidualUNet(base_channels=4)
        for value in [float("nan"), float("inf"), -0.1, 1.1]:
            with self.assertRaises(ValueError):
                model(torch.full((1, 3, 16, 16), value))

    def test_ssim_matches_independent_double_precision_convolution(self):
        rng = np.random.default_rng(123)
        a, b = rng.random((1, 3, 19, 21)), rng.random((1, 3, 19, 21))
        g = np.exp(-np.arange(-5, 6, dtype=float) ** 2 / (2 * 1.5 ** 2))
        kernel = np.outer(g / g.sum(), g / g.sum())
        channels = []
        for x, y in zip(a[0], b[0]):
            mx, my, xx, yy, xy = [convolve2d(v, kernel, mode="valid") for v in [x, y, x*x, y*y, x*y]]
            expected = ((2*mx*my+.01**2) * (2*(xy-mx*my)+.03**2) /
                        ((mx*mx+my*my+.01**2) * (xx-mx*mx+yy-my*my+.03**2)))
            channels.append(expected.mean())
        self.assertAlmostEqual(ssim(torch.from_numpy(a), torch.from_numpy(b)).item(), np.mean(channels), places=12)

    def test_metric_identities_constants_and_rejections(self):
        a, b = torch.zeros(2, 3, 16, 17), torch.ones(2, 3, 16, 17)
        torch.testing.assert_close(ssim(a, a), torch.ones(2))
        torch.testing.assert_close(ssim(a, b), torch.full((2,), .0001 / 1.0001))
        torch.testing.assert_close(psnr(a, b), torch.zeros(2))
        self.assertTrue(torch.isinf(psnr(a, a)).all())
        with self.assertRaises(ValueError):
            ssim(a[..., :10], b[..., :10])

    def test_l1_and_ssim_backpropagate_and_update_prediction(self):
        torch.manual_seed(1)
        model = ResidualUNet(base_channels=4)
        x, target = torch.full((1, 3, 16, 16), .4), torch.full((1, 3, 16, 16), .6)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        y = model(x)
        loss = (y-target).abs().mean() + .2*(1-ssim(y, target).mean())
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        optimizer.step()
        self.assertGreater((model(x)-x).abs().max().item(), 0)

    def test_loss_policy_and_smoke_cap_enforced(self):
        config = json.loads((ROOT / "configs/rgb_baseline.json").read_text())
        validate_config(config)
        for key, value in [("adversarial", .01), ("ssim", 0), ("l1", float("nan"))]:
            invalid = deepcopy(config)
            invalid["loss_weights"][key] = value
            with self.assertRaises(ValueError):
                validate_config(invalid)
        with self.assertRaises(ValueError):
            validate_config(config | {"max_steps": 2})

    def test_centered_reference_identity_preserves_mean_shift_and_penalizes_spatial_change(self):
        from aurora.training.run import reference_identity_loss
        reference = torch.zeros(2, 3, 11, 13)
        shift = torch.tensor(.125, requires_grad=True)
        raw = reference + shift
        self.assertEqual(reference_identity_loss(raw, reference).item(), .125)
        loss = reference_identity_loss(raw, reference, "centered_residual")
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertEqual(shift.grad.item(), 0)
        pattern = torch.linspace(-1, 1, 143).reshape(1, 1, 11, 13).expand_as(reference)
        amplitude = torch.tensor(.125, requires_grad=True)
        centered = reference_identity_loss(reference+.125+amplitude*pattern, reference, "centered_residual")
        centered.backward()
        self.assertGreater(amplitude.grad.item(), 0)
        self.assertAlmostEqual(centered.item(), .125*pattern.abs().mean().item(), places=6)
        with self.assertRaises(ValueError):
            reference_identity_loss(raw, reference, "unknown")

    def test_valid_center_loss_masks_gradients_and_preserves_raw_l1(self):
        from aurora.training.run import reconstruction_terms
        raw = torch.full((1, 3, 144, 144), 1.25, requires_grad=True)
        target = torch.full_like(raw, .5)
        l1, structural = reconstruction_terms(raw, raw.clamp(0, 1), target, 64)
        self.assertEqual(l1.item(), .75)
        (l1+.2*structural).backward()
        self.assertGreater(raw.grad[..., 64:80, 64:80].abs().min().item(), 0)
        border = raw.grad.clone(); border[..., 64:80, 64:80] = 0
        self.assertEqual(border.abs().max().item(), 0)
        a, b = torch.rand(1, 3, 16, 16), torch.rand(1, 3, 16, 16)
        full_l1, full_structural = reconstruction_terms(a, a, b)
        self.assertTrue(torch.equal(full_l1, (a-b).abs().mean()))
        self.assertTrue(torch.equal(full_structural, 1-ssim(a,b).mean()))
        for invalid in [-1, True, 1.5, 3]:
            with self.assertRaises(ValueError):
                reconstruction_terms(a,a,b,invalid)
        cfg = json.loads((ROOT / "configs/rgb_baseline.json").read_text())
        with self.assertRaisesRegex(ValueError, "restricted"):
            validate_config(cfg | {"loss_margin":64})

    def fixture_run(self, name):
        raw, ref = self.root / "raw.png", self.root / "reference.png"
        Image.new("RGB", (19, 17), (70, 100, 140)).save(raw)
        Image.new("RGB", (19, 17), (90, 120, 150)).save(ref)
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps({"status": "frozen", "pairs": [{"id": "fixture", "scene_id": "scene",
            "split": "train", "raw_path": raw.name, "reference_path": ref.name,
            "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
            "reference_sha256": hashlib.sha256(ref.read_bytes()).hexdigest()}], "challenging": []}))
        digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
        manifest.with_suffix(".json.sha256").write_text(digest)
        config = json.loads((ROOT / "configs/rgb_smoke.json").read_text()) | {
            "manifest": str(manifest), "manifest_sha256": digest, "data_root": str(self.root),
            "model": {"base_channels": 4, "residual_scale": .5}, "crop_size": 16,
            "batch_size": 1, "epochs": 2, "max_steps": 2, "threads": 2, "output": str(self.root / name)}
        path = self.root / f"{name}.json"
        path.write_text(json.dumps(config))
        return path

    def test_seed_reproducibility_reload_export_and_corruption(self):
        a = self.fixture_run("first")
        first = train(a)
        second = train(self.fixture_run("second"))
        self.assertTrue(first["checkpoint_reload_exact"])
        self.assertEqual(first["mean_training_loss"], second["mean_training_loss"])
        original = self.root / "first/last/bundle.json"
        model, other = RGBInference(original), RGBInference(self.root / "second/last/bundle.json")
        for key, value in model.model.state_dict().items():
            self.assertTrue(torch.equal(value, other.model.state_dict()[key]))
        valid = validate_image(self.root / "raw.png")
        result, report = model.enhance(valid)
        self.assertEqual(result.size, valid.image.size)
        self.assertFalse(report["selected"])
        export = export_png(result, self.root / "enhanced.png")
        with Image.open(export) as reopened:
            self.assertEqual(reopened.tobytes(), result.tobytes())
            self.assertFalse(reopened.getexif())
            self.assertEqual(set(reopened.info), {"icc_profile"})
            self.assertIn("sRGB", ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(BytesIO(reopened.info["icc_profile"]))))
        with self.assertRaises(FileExistsError):
            export_png(result, export)
        with self.assertRaises(FileExistsError):
            train(a)
        (original.parent / "weights.pt").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "checksum"):
            RGBInference(original)

    def test_unclamped_l1_restores_saturation_gradient(self):
        model = ResidualUNet(base_channels=4)
        with torch.no_grad():
            model.head.bias.fill_(2)
        x, target = torch.full((1, 3, 16, 16), .8), torch.full((1, 3, 16, 16), .7)
        bounded = model(x)
        self.assertTrue((bounded == 1).all())
        (bounded-target).abs().mean().backward()
        self.assertTrue((model.head.bias.grad == 0).all())
        model.zero_grad(set_to_none=True)
        raw = model.forward_unclamped(x)
        loss = (raw-target).abs().mean() + .2 * (1-ssim(raw.clamp(0, 1), target).mean())
        loss.backward()
        self.assertTrue((model.head.bias.grad > 0).all())
        self.assertTrue(torch.isfinite(model.head.bias.grad).all())

    def test_unclamped_training_configuration_and_reload(self):
        path = self.fixture_run("unclamped")
        config = json.loads(path.read_text()) | {"l1_prediction": "unclamped"}
        path.write_text(json.dumps(config))
        result = train(path)
        self.assertTrue(result["finite_losses"] and result["checkpoint_reload_exact"])
        bundle = RGBInference(self.root / "unclamped/last/bundle.json")
        self.assertEqual(bundle.bundle["configuration"]["l1_prediction"], "unclamped")
        self.assertEqual(bundle.bundle["provenance"]["l1_prediction"], "unclamped")
        with self.assertRaisesRegex(ValueError, "l1_prediction"):
            validate_config(config | {"l1_prediction": "invalid"})


if __name__ == "__main__":
    unittest.main()
