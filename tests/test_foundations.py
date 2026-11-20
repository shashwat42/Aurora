from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import json
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageCms
import torch

from aurora.app import preview_image
from aurora.dataset import PairedDataset, synchronized_geometry
from aurora.depth import DepthEstimator, normalize_depth
from aurora.image import validate_image, ImageValidationError
from aurora.manifest import read_manifest, development_rows, safe_path

ROOT = Path(__file__).resolve().parents[1]


class Foundations(unittest.TestCase):
    def setUp(self):
        (ROOT / "artifacts/tests").mkdir(exist_ok=True, parents=True)
        self.temp = TemporaryDirectory(dir=ROOT / "artifacts/tests")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def image(self, name="image.png", mode="RGB", size=(31, 17), **kwargs):
        path = self.root / name
        Image.new(mode, size).save(path, **kwargs)
        return path

    def manifest(self, **updates):
        raw = self.image("raw.png", size=(32, 32))
        ref = self.image("ref.png", size=(32, 32))
        document = {"status": "frozen", "pairs": [{"id": "a", "scene_id": "s", "split": "train",
                   "raw_path": "raw.png", "reference_path": "ref.png",
                   "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                   "reference_sha256": hashlib.sha256(ref.read_bytes()).hexdigest()}], "challenging": []} | updates
        path = self.root / "manifest.json"
        path.write_text(json.dumps(document))
        path.with_suffix(".json.sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest())
        return path

    def model(self, **overrides):
        (self.root / "configs").mkdir(exist_ok=True)
        (self.root / "weights").mkdir(exist_ok=True)
        config = {"local_directory": "weights", "model_id": "fixture", "revision": "pinned",
                  "files_sha256": {}, "processor_configuration": {}, "preprocessing_version": "v1"} | overrides
        path = self.root / "configs/depth.json"
        path.write_text(json.dumps(config))
        return DepthEstimator(path)

    def test_native_geometry_and_hash(self):
        p = self.image()
        valid = validate_image(p)
        self.assertEqual(valid.image.size, (31, 17))
        self.assertEqual(valid.source_sha256, hashlib.sha256(p.read_bytes()).hexdigest())

    def test_exif_orientation_and_metadata_removal(self):
        exif = Image.Exif()
        exif[274] = 6
        p = self.image("oriented.jpg", exif=exif)
        valid = validate_image(p)
        self.assertEqual(valid.image.size, (17, 31))
        self.assertFalse(valid.image.getexif())

    def test_valid_srgb_profile(self):
        profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        self.assertEqual(validate_image(self.image(icc_profile=profile)).image.mode, "RGB")

    def test_invalid_icc_rejected(self):
        with self.assertRaisesRegex(ImageValidationError, "ICC"):
            validate_image(self.image(icc_profile=b"broken"))

    def test_corrupt_input_clears_preview(self):
        p = self.root / "corrupt.png"
        p.write_bytes(b"not an image")
        image, message = preview_image(p)
        self.assertIsNone(image)
        self.assertIn("Cannot preview", message)

    def test_wrong_modes_rejected(self):
        for mode in ["L", "RGBA", "I;16"]:
            with self.subTest(mode=mode), self.assertRaises(ImageValidationError):
                validate_image(self.image(mode=mode))

    def test_byte_limit(self):
        with self.assertRaisesRegex(ImageValidationError, "byte"):
            validate_image(self.image(), max_bytes=1)

    def test_pixel_limit(self):
        with self.assertRaisesRegex(ImageValidationError, "pixel"):
            validate_image(self.image(), max_pixels=10)

    def test_animated_png_rejected(self):
        p = self.root / "animated.png"
        Image.new("RGB", (20, 20), "red").save(p, save_all=True, append_images=[Image.new("RGB", (20, 20), "blue")])
        with self.assertRaisesRegex(ImageValidationError, "multi-frame"):
            validate_image(p)

    def test_mpo_rejected_despite_png_extension(self):
        # Actual vendor file: checks encoding only; never uses it to tune enhancement.
        with self.assertRaisesRegex(ImageValidationError, "MPO"):
            validate_image(ROOT / "data/uieb/challenging/2.png")

    def test_empty_upload_resets(self):
        self.assertIsNone(preview_image(None)[0])

    def test_explicit_preview_mode_keeps_enhancement_disabled(self):
        from aurora.app import create_app
        app = create_app(preview_only=True)
        self.addCleanup(app.aurora_manager.close)
        config = app.get_config_file()
        enhance = [c for c in config["components"] if c["type"] == "button" and "Enhance" in c["props"].get("value", "")]
        self.assertEqual(len(enhance), 1)
        self.assertFalse(enhance[0]["props"]["interactive"])
        preview = [c for c in config["components"] if c["type"] == "image"][0]
        self.assertEqual(preview["props"]["buttons"], ["fullscreen"])

    def test_synchronized_crop_flip_rotation_and_small_padding(self):
        a = np.arange(15 * 19, dtype=np.float32).reshape(15, 19)
        rgb = np.stack([a] * 3, axis=-1)
        for seed in range(8):
            out = synchronized_geometry([rgb, rgb + 1, a], crop_size=32, seed=seed)
            np.testing.assert_array_equal(out[0][..., 0], out[2])
            np.testing.assert_array_equal(out[0] + 1, out[1])
            self.assertEqual(out[2].shape, (32, 32))

    def test_geometry_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "aligned"):
            synchronized_geometry([np.zeros((3, 4)), np.zeros((4, 3))], crop_size=2, seed=1)

    def test_constant_and_nonfinite_depth(self):
        np.testing.assert_array_equal(normalize_depth(np.ones((3, 4))), np.zeros((3, 4)))
        for value in [float("inf"), float("nan")]:
            with self.assertRaises(ValueError):
                normalize_depth(np.array([[value]]))

    def test_depth_cache_roundtrip_and_tampering(self):
        model = self.model()
        valid = validate_image(self.image())
        values = np.linspace(0, 1, valid.image.width * valid.image.height, dtype=np.float32).reshape(valid.image.height, valid.image.width)
        with patch.object(model, "predict", return_value=(values, {"seconds": 0.0})):
            model.cache(valid)
        np.testing.assert_array_equal(model.load_cached(valid), values)
        path, _ = model._paths(valid)
        path.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "checksum"):
            model.load_cached(valid)

    def test_cache_keys_change_with_source_and_preprocessing(self):
        model = self.model()
        first = validate_image(self.image("first.png"))
        second = validate_image(self.image("second.png", size=(5, 8)))
        self.assertNotEqual(model.cache_key(first), model.cache_key(second))
        other = self.model(preprocessing_version="v2")
        self.assertNotEqual(model.cache_key(first), other.cache_key(first))

    def test_depth_artifact_tampering(self):
        with self.assertRaisesRegex(ValueError, "checksum"):
            (self.root / "configs").mkdir()
            (self.root / "weights").mkdir()
            (self.root / "weights/fake").write_text("different")
            p = self.root / "configs/depth.json"
            p.write_text(json.dumps({"local_directory": "weights", "files_sha256": {"fake": "bad"}}))
            DepthEstimator(p)

    def test_holdouts_rejected(self):
        for split in ["test", "challenging"]:
            with self.assertRaisesRegex(ValueError, "sealed"):
                development_rows({"pairs": []}, split)

    def test_scene_leakage_rejected(self):
        p = self.manifest(pairs=[{"id": "a", "scene_id": "same", "split": "train"},
                                 {"id": "b", "scene_id": "same", "split": "test"}])
        with self.assertRaisesRegex(ValueError, "scene"):
            read_manifest(p)

    def test_inspected_development_image_cannot_enter_holdout(self):
        for split in ["selection", "test"]:
            p = self.manifest(protected_development_ids=["a"], pairs=[{"id": "a", "scene_id": "s", "split": split}])
            with self.assertRaisesRegex(ValueError, "inspected"):
                read_manifest(p)

    def test_decoded_duplicate_cannot_cross_splits_or_challenging(self):
        pairs = [{"id": "a", "scene_id": "one", "split": "train", "raw_rgb_sha256": "same"},
                 {"id": "b", "scene_id": "two", "split": "test", "raw_rgb_sha256": "same"}]
        with self.assertRaisesRegex(ValueError, "identical"):
            read_manifest(self.manifest(pairs=pairs))
        with self.assertRaisesRegex(ValueError, "identical"):
            read_manifest(self.manifest(pairs=pairs[:1], challenging=[{"id": "x", "scene_id": "other", "raw_rgb_sha256": "same"}]))

    def test_changed_and_draft_manifests_rejected(self):
        p = self.manifest()
        p.write_text(p.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "frozen"):
            read_manifest(p)
        p = self.manifest(status="draft")
        with self.assertRaisesRegex(ValueError, "frozen"):
            read_manifest(p)

    def test_dataset_determinism_native_evaluation_and_tampering(self):
        p = self.manifest()
        data = PairedDataset(self.root, p, crop_size=16)
        self.assertTrue(torch.equal(data[0]["input"], data[0]["input"]))
        with self.assertRaisesRegex(ValueError, "native"):
            PairedDataset(self.root, p, split="diagnostic", crop_size=16)
        Image.new("RGB", (32, 32), "white").save(self.root / "raw.png")
        with self.assertRaisesRegex(ValueError, "changed"):
            data[0]

    def test_dataset_path_escape_rejected(self):
        with self.assertRaisesRegex(ValueError, "escapes"):
            safe_path(self.root, "../secret")


if __name__ == "__main__":
    unittest.main()
