"""Frozen, offline depth inference and content-addressed, verified float32 caches."""
from pathlib import Path
import hashlib
import json
import os
import tempfile
import time

import numpy as np
import torch
from torch.nn import functional as F

from .image import PREPROCESS_VERSION
from .manifest import file_hash


def normalize_depth(values):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Depth must be a finite two-dimensional map.")
    lo, hi = float(values.min()), float(values.max())
    return np.zeros_like(values) if hi - lo <= 1e-6 else (values - lo) / (hi - lo)


class DepthEstimator:
    def __init__(self, config_path, *, cache_root=None):
        self.config_path = Path(config_path).resolve()
        self.config = json.loads(self.config_path.read_text())
        self.root = self.config_path.parent.parent
        self.directory = self.root / self.config["local_directory"]
        for name, expected in self.config["files_sha256"].items():
            if file_hash(self.directory / name) != expected:
                raise ValueError(f"Depth artifact checksum mismatch: {name}")
        fingerprint_data = {"model_id": self.config["model_id"], "revision": self.config["revision"],
                            "files": self.config["files_sha256"], "image_preprocessing": PREPROCESS_VERSION,
                            "depth_preprocessing": self.config["preprocessing_version"],
                            "processor": self.config["processor_configuration"]}
        self.fingerprint = hashlib.sha256(json.dumps(fingerprint_data, sort_keys=True).encode()).hexdigest()
        self.cache_root = Path(cache_root) if cache_root else self.root / "data/depth" / self.fingerprint
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.processor = self.model = None

    def _load(self):
        if self.model is None:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            torch.set_num_threads(self.config.get("threads", 4))
            self.processor = AutoImageProcessor.from_pretrained(self.directory, local_files_only=True, use_fast=False)
            self.model = AutoModelForDepthEstimation.from_pretrained(self.directory, local_files_only=True, use_safetensors=True)
            self.model.to(self.config["device"]).eval().requires_grad_(False)

    def cache_key(self, valid):
        return hashlib.sha256(f"{valid.source_sha256}:{valid.image.size}:{self.fingerprint}".encode()).hexdigest()

    def predict(self, valid):
        self._load()
        start = time.perf_counter()
        inputs = self.processor(images=valid.image, return_tensors="pt")
        internal_hw = list(inputs["pixel_values"].shape[-2:])
        inputs = {k: v.to(self.config["device"]) for k, v in inputs.items()}
        with torch.inference_mode():
            prediction = self.model(**inputs).predicted_depth
            if not torch.isfinite(prediction).all():
                raise ValueError("Estimator produced nonfinite depth.")
            w, h = valid.image.size
            # The selected processor resizes only (no crop/pad). Restore its inverse geometry.
            restored = F.interpolate(prediction.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False)
        array = normalize_depth(restored[0, 0].float().cpu().numpy())
        return array, {"internal_hw": internal_hw, "native_hw": [h, w],
                       "seconds": time.perf_counter() - start, "min": float(array.min()), "max": float(array.max())}

    def _paths(self, valid):
        key = self.cache_key(valid)
        return self.cache_root / f"{key}.npy", self.cache_root / f"{key}.json"

    def load_cached(self, valid):
        path, meta_path = self._paths(valid)
        meta = json.loads(meta_path.read_text())
        if meta["input_sha256"] != valid.source_sha256 or meta["estimator_fingerprint"] != self.fingerprint:
            raise ValueError("Depth cache belongs to another input or estimator.")
        if file_hash(path) != meta["map_sha256"]:
            raise ValueError("Depth cache checksum mismatch.")
        array = np.load(path, allow_pickle=False)
        if array.dtype != np.float32 or array.shape != (valid.image.height, valid.image.width):
            raise ValueError("Depth cache geometry or dtype mismatch.")
        if not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
            raise ValueError("Depth cache is outside finite [0,1].")
        return array

    def cache(self, valid):
        path, meta_path = self._paths(valid)
        if path.exists() and meta_path.exists():
            array = self.load_cached(valid)
            return array, json.loads(meta_path.read_text())
        array, meta = self.predict(valid)
        fd, temporary = tempfile.mkstemp(dir=self.cache_root, suffix=".npy")
        try:
            with os.fdopen(fd, "wb") as stream:
                np.save(stream, array, allow_pickle=False)
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
        meta.update(input_sha256=valid.source_sha256, estimator_fingerprint=self.fingerprint,
                    map_sha256=file_hash(path), cache_key=self.cache_key(valid), relative_not_metric=True)
        temp_meta = meta_path.with_suffix(".json.tmp")
        temp_meta.write_text(json.dumps(meta, indent=2) + "\n")
        os.replace(temp_meta, meta_path)
        return array, meta
