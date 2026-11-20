"""Paired RGB/depth loading with deterministic synchronized geometric transforms."""
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .image import validate_image
from .manifest import development_rows, read_manifest, safe_path


def synchronized_geometry(arrays, *, crop_size, seed):
    shapes = [a.shape[:2] for a in arrays]
    if len(set(shapes)) != 1:
        raise ValueError("RGB/reference/depth shapes are not aligned.")
    h, w = shapes[0]
    if min(h, w) < crop_size:
        # Training-only reflection padding, shared by RGB, reference and cached depth.
        # Native-size validation/inference uses crop_size=None and never follows this path.
        padded = []
        for a in arrays:
            pad = [(0, max(0, crop_size - h)), (0, max(0, crop_size - w))]
            pad += [(0, 0)] * (a.ndim - 2)
            padded.append(np.pad(a, pad, mode="reflect"))
        arrays = padded
        h, w = arrays[0].shape[:2]
    rng = np.random.default_rng(seed)
    y = int(rng.integers(h - crop_size + 1))
    x = int(rng.integers(w - crop_size + 1))
    flip_x, flip_y = rng.integers(2, size=2)
    turns = int(rng.integers(4))
    results = []
    for a in arrays:
        a = a[y:y + crop_size, x:x + crop_size]
        if flip_x:
            a = np.flip(a, axis=1)
        if flip_y:
            a = np.flip(a, axis=0)
        results.append(np.ascontiguousarray(np.rot90(a, turns)))
    return results


class PairedDataset(Dataset):
    def __init__(self, data_root, manifest, split="train", *, crop_size=None, seed=20261101, depth=None,
                 include_scene_stats=False):
        self.root = Path(data_root)
        self.rows = development_rows(read_manifest(manifest), split)
        self.split, self.crop_size, self.seed, self.depth = split, crop_size, seed, depth
        self.include_scene_stats = include_scene_stats
        if split != "train" and crop_size is not None:
            raise ValueError("Development evaluation requires native dimensions, with crop_size=None.")
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        raw = validate_image(safe_path(self.root, row["raw_path"]))
        ref = validate_image(safe_path(self.root, row["reference_path"]))
        if raw.source_sha256 != row["raw_sha256"] or ref.source_sha256 != row["reference_sha256"]:
            raise ValueError("Dataset bytes changed after manifest freeze.")
        arrays = [np.asarray(raw.image, dtype=np.float32) / 255, np.asarray(ref.image, dtype=np.float32) / 255]
        if arrays[0].shape != arrays[1].shape:
            raise ValueError("Oriented RGB/reference dimensions differ.")
        stats = None
        if self.include_scene_stats:
            from .models.scene_rgb import scene_statistics
            native = torch.from_numpy(arrays[0].transpose(2, 0, 1).copy()).unsqueeze(0)
            stats = scene_statistics(native)[0]
        if self.depth is not None:
            arrays.append(self.depth.load_cached(raw))
        if self.crop_size:
            token = f"{self.seed}:{self.epoch}:{row['id']}".encode()
            seed = int.from_bytes(hashlib.sha256(token).digest()[:8], "little")
            arrays = synchronized_geometry(arrays, crop_size=self.crop_size, seed=seed)
        def tensor(a):
            if a.ndim == 2:
                a = a[..., None]
            return torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))
        batch = {"id": row["id"], "input": tensor(arrays[0]), "reference": tensor(arrays[1])}
        if stats is not None:
            batch['scene_stats'] = stats
        if len(arrays) == 3:
            batch["depth"] = tensor(arrays[2])
        return batch
