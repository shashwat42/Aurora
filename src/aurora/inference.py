"""Shared native-size RGB inference and metadata-clean PNG export."""
from pathlib import Path
import json
import time

import numpy as np
from PIL import Image, ImageCms
import torch

from .image import PREPROCESS_VERSION
from .manifest import file_hash, safe_path, read_manifest
from .models import ARCHITECTURES, SCENE_ARCHITECTURE, DEPTH_ARCHITECTURE, DEPTH_ARCHITECTURES, build_rgb


class RGBInference:
    def __init__(self, bundle_path, *, depth_policy='prefer_cache'):
        if depth_policy not in {'live', 'prefer_cache'}:
            raise ValueError('Unsupported depth inference policy.')
        self.depth_policy = depth_policy
        self.last_depth_mode = None
        self.path = Path(bundle_path)
        self.bundle = json.loads(self.path.read_text())
        b = self.bundle
        if b.get("schema_version") != 1 or b.get("architecture") not in ARCHITECTURES or b.get("preprocessing_version") != PREPROCESS_VERSION:
            raise ValueError("Incompatible model bundle; use a matching Aurora RGB checkpoint.")
        if b['configuration']['architecture'] != b['architecture'] or b['configuration']['model'] != b['model']:
            raise ValueError('Bundle architecture/model differs from configuration.')
        if b['architecture'] in ({SCENE_ARCHITECTURE} | DEPTH_ARCHITECTURES) and b['configuration'].get('scene_context') != 'native-input-channel-moments-v1':
            raise ValueError('Scene checkpoint requires native input moments context.')
        if type(b.get("training_steps")) is not int or b["training_steps"] < 1:
            raise ValueError("The bundle does not contain a trained checkpoint.")
        weights = safe_path(self.path.parent, b["weights"])
        if file_hash(weights) != b["weights_sha256"]:
            raise ValueError("Model weights checksum mismatch; restore the original checkpoint.")
        self.model = build_rgb(b['architecture'], b['model'])
        self.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
        if not all(torch.isfinite(p).all() for p in self.model.parameters()):
            raise ValueError("Model checkpoint contains non-finite weights.")
        self.model.eval().requires_grad_(False)
        self.depth = None
        if b['architecture'] in DEPTH_ARCHITECTURES:
            from .depth import DepthEstimator
            config = b['configuration']
            if file_hash(config['depth_config']) != config['depth_config_sha256']:
                raise ValueError('Depth configuration identity changed.')
            self.depth = DepthEstimator(config['depth_config'])
            if self.depth.fingerprint != config['depth_fingerprint']:
                raise ValueError('Depth estimator fingerprint changed.')

    def predict(self, valid, *, stage_callback=None):
        """Native float RGB; shared by evaluation, CLI and worker inference."""
        manifest_path = self.bundle["configuration"]["manifest"]
        manifest = read_manifest(manifest_path)
        if file_hash(manifest_path) != self.bundle["configuration"]["manifest_sha256"]:
            raise ValueError("Model and dataset manifest identities differ.")
        sealed = [r for r in manifest["pairs"] if r["split"] == "test"] + manifest.get("challenging", [])
        if valid.source_sha256 in {r["raw_sha256"] for r in sealed}:
            raise ValueError("Final test and challenging inputs remain sealed during development.")
        array = np.array(valid.image, dtype=np.float32) / 255
        x = torch.from_numpy(array.transpose(2, 0, 1).copy()).unsqueeze(0)
        context = {}
        if self.depth is not None:
            if stage_callback:stage_callback('depth')
            paths = self.depth._paths(valid)
            if self.depth_policy == 'prefer_cache' and any(p.exists() for p in paths):
                depth = self.depth.load_cached(valid)  # Corrupt/partial caches fail; never RGB fallback.
                self.last_depth_mode = 'cached'
            else:
                depth, _ = self.depth.predict(valid)
                self.last_depth_mode = 'live'
            context['depth'] = torch.from_numpy(depth.copy()).unsqueeze(0).unsqueeze(0)
        if stage_callback:stage_callback('processing')
        with torch.inference_mode():
            y = self.model(x, **context)
        if y.shape != x.shape or not torch.isfinite(y).all() or (y < 0).any() or (y > 1).any():
            raise ValueError("Enhancement produced invalid pixels or dimensions.")
        return y

    def enhance(self, valid, *, stage_callback=None):
        started = time.perf_counter()
        y = self.predict(valid, stage_callback=stage_callback)
        pixels = np.rint(y[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        image = Image.fromarray(pixels)
        return image, {"seconds": time.perf_counter() - started, "source_sha256": valid.source_sha256,
                       "weights_sha256": self.bundle["weights_sha256"],
                       "mode": "RGB-depth experimental prototype" if self.depth else "RGB prototype",
                       "depth_fingerprint": self.depth.fingerprint if self.depth else None,
                       "depth_mode": self.last_depth_mode,
                       "run_kind": self.bundle["run_kind"], "selected": self.bundle["selected"],
                       "width": image.width, "height": image.height}


def export_png(image, path):
    path = Path(path)
    if path.suffix.lower() != ".png":
        raise ValueError("Enhanced output must use a .png filename.")
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = Image.frombytes("RGB", image.size, image.tobytes())
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    # Exclusive creation protects original photos and prior results from overwrites.
    with path.open("xb") as stream:
        try:
            clean.save(stream, format="PNG", icc_profile=profile)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
    return path
