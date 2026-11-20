"""Manifest integrity and holdout isolation. No final-test access in Phase 1."""
import hashlib
import json
from pathlib import Path

DEVELOPMENT_SPLITS = {"train", "diagnostic", "selection"}


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_manifest(path, *, require_frozen=True):
    path = Path(path)
    document = json.loads(path.read_text())
    if require_frozen:
        expected = path.with_suffix(path.suffix + ".sha256").read_text().split()[0]
        if file_hash(path) != expected or document["status"] != "frozen":
            raise ValueError("Manifest is not frozen or its checksum has changed.")
    seen, scene_splits, pixel_splits = set(), {}, {}
    protected = set(document.get("protected_development_ids", []))
    for row in document["pairs"]:
        if row["id"] in seen:
            raise ValueError("Duplicate pair ID in manifest.")
        seen.add(row["id"])
        split = row["split"]
        if split not in DEVELOPMENT_SPLITS | {"test"}:
            raise ValueError("Unknown split.")
        if row["id"] in protected and split not in {"train", "diagnostic"}:
            raise ValueError("A previously inspected development image entered a holdout.")
        prior = scene_splits.setdefault(row["scene_id"], split)
        if prior != split:
            raise ValueError("A scene crosses split boundaries.")
        digest = row.get("raw_rgb_sha256")
        if digest and pixel_splits.setdefault(digest, split) != split:
            raise ValueError("An identical decoded input crosses split boundaries.")
    for row in document.get("challenging", []):
        prior = scene_splits.setdefault(row["scene_id"], "challenging")
        if prior != "challenging":
            raise ValueError("A challenging scene crosses into paired data.")
        digest = row.get("raw_rgb_sha256")
        if digest and pixel_splits.setdefault(digest, "challenging") != "challenging":
            raise ValueError("An identical decoded input crosses into challenging data.")
    return document


def safe_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Dataset path escapes data root.")
    return path


def development_rows(manifest, split):
    if split not in DEVELOPMENT_SPLITS:
        raise ValueError("Final test and challenging sets are sealed until release candidates are frozen.")
    return [row for row in manifest["pairs"] if row["split"] == split]
