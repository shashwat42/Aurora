"""Check all prepared caches and save one real synchronized batch demonstration."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from aurora.dataset import synchronized_geometry
from aurora.depth import DepthEstimator
from aurora.image import validate_image
from aurora.manifest import read_manifest, file_hash

parser = argparse.ArgumentParser()
parser.add_argument("--manifest", default="data/metadata/manifest.v1.json")
args = parser.parse_args()
manifest_path = ROOT / args.manifest
manifest = read_manifest(manifest_path)
model = DepthEstimator(ROOT / "configs/depth.json")
rows = [r for r in manifest["pairs"] if r["split"] in {"train", "diagnostic"}]
verified = []
for row in rows:
    valid = validate_image(ROOT / "data" / row["raw_path"])
    if valid.source_sha256 != row["raw_sha256"]:
        raise ValueError("Input changed after manifest freeze")
    model.load_cached(valid)
    verified.append(row["id"])
    if len(verified) % 100 == 0:
        print(f"Verified {len(verified)}/{len(rows)} cache entries", flush=True)

training = [r for r in rows if r["split"] == "train"]
small = next(r for r in training if min(r["width"], r["height"]) < 256)
sample_rows = training[:3] + [small]
sheet = Image.new("RGB", (3 * 256, 4 * 280), "#20252b")
draw = ImageDraw.Draw(sheet)
samples = []
for i, row in enumerate(sample_rows):
    raw = validate_image(ROOT / "data" / row["raw_path"])
    reference = validate_image(ROOT / "data" / row["reference_path"])
    if reference.source_sha256 != row["reference_sha256"]:
        raise ValueError("Reference changed after manifest freeze")
    depth = model.load_cached(raw)
    arrays = [np.asarray(raw.image, dtype=np.float32) / 255,
              np.asarray(reference.image, dtype=np.float32) / 255, depth]
    transformed = synchronized_geometry(arrays, crop_size=256, seed=20261101 + i)
    for j, array in enumerate(transformed):
        assert np.isfinite(array).all() and 0 <= array.min() <= array.max() <= 1
        im = Image.fromarray(np.rint(array * 255).astype(np.uint8)).convert("RGB")
        sheet.paste(im, (j * 256, i * 280 + 24))
    draw.text((3, i * 280 + 3), row["id"] + "   input / reference / relative depth", fill="white")
    samples.append({"id": row["id"], "native_wh": list(raw.image.size), "crop_hw": [256, 256],
                    "reflection_padding_required": min(raw.image.size) < 256})
sheet.save(ROOT / "artifacts/data_review/synchronized_batch.jpg", quality=92)
report = {"manifest_status": manifest["status"], "manifest_sha256": file_hash(manifest_path),
          "verified_train_and_diagnostic_maps": len(verified), "verified_ids": verified,
          "samples": samples, "photometric_augmentation": False,
          "note": "Verified cache coverage and synchronized samples for the immutable frozen manifest. No training or final-test evaluation."}
(ROOT / "artifacts/data_review/cache_verification.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k != "verified_ids"}, indent=2))
