"""Diagnostic counterfactuals from cached residuals; no training/model forwards."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import gaussian_filter
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aurora.image import validate_image
from aurora.inference import export_png
from aurora.manifest import file_hash, read_manifest, safe_path
from aurora.training.metrics import METRIC_VERSION, psnr, ssim


VARIANTS = ["original", "spatial_mean", "sigma8", "sigma32"]


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def tensor(array):
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1), dtype=np.float32)).unsqueeze(0)


def process(identifier, output_root, *, variants=VARIANTS, correction_transform=None, method_metadata=None):
    started = time.perf_counter()
    manifest_path = ROOT / "data/metadata/manifest.v1.json"
    manifest = read_manifest(manifest_path)
    row = next(r for r in manifest["pairs"] if r["id"] == identifier)
    if row["split"] != "diagnostic":
        raise ValueError("Counterfactual ablations are restricted to diagnostic images.")
    source = ROOT / "artifacts/phase2/diagnosis_v2/residual_probe"
    provenance = json.loads((source / "report.json").read_text())
    if provenance["manifest_sha256"] != file_hash(manifest_path):
        raise ValueError("Cached residual manifest identity differs.")
    bundle_dir = ROOT / "artifacts/phase2/rgb_baseline_v2/epoch-020"
    bundle = json.loads((bundle_dir / "bundle.json").read_text())
    if bundle["weights_sha256"] != provenance["weights_sha256"] or file_hash(bundle_dir / bundle["weights"]) != provenance["weights_sha256"]:
        raise ValueError("Cached residual checkpoint identity differs.")
    raw = validate_image(safe_path(ROOT / "data", row["raw_path"]))
    reference = validate_image(safe_path(ROOT / "data", row["reference_path"]))
    if raw.source_sha256 != row["raw_sha256"] or reference.source_sha256 != row["reference_sha256"]:
        raise ValueError("Source/reference bytes differ from the frozen manifest.")
    image = np.asarray(raw.image, dtype=np.float32) / 255
    target = np.asarray(reference.image, dtype=np.float32) / 255
    npz = source / f"{identifier}_residual.npz"
    with np.load(npz, allow_pickle=False) as arrays:
        residual, unclamped = arrays["residual"], arrays["unclamped"]
    if residual.shape != image.shape or target.shape != image.shape or not np.isfinite(residual).all() or not np.isfinite(unclamped).all():
        raise ValueError("Cached arrays are invalid or misaligned.")
    if not np.allclose(image + residual, unclamped, atol=1e-7, rtol=0):
        raise ValueError("Residual does not reconstruct its cached pre-clamp prediction.")
    cached_path = ROOT / f"artifacts/phase2/evaluation_v2/epoch-020/diagnostic/images/{identifier}.png"
    summary = json.loads((cached_path.parent.parent / "summary.json").read_text())
    cached_record = next(r for r in summary["records"] if r["id"] == identifier)
    if file_hash(cached_path) != cached_record["png_sha256"]:
        raise ValueError("Cached baseline PNG checksum differs.")
    with Image.open(cached_path) as original:
        if not np.array_equal(np.rint(unclamped.clip(0, 1)*255).astype(np.uint8), np.asarray(original)):
            raise ValueError("Original residual control does not reproduce the cached PNG.")

    output = output_root / identifier
    output.mkdir(parents=True, exist_ok=False)
    mean = residual.mean(axis=(0, 1), dtype=np.float64)
    centered = (residual - mean).astype(np.float32)
    roi = provenance["images"][identifier]["contour_roi"]["xywh"]
    x, y, width, height = roi
    reference_tensor = tensor(target)
    records, previews = [], [("Input", raw.image), ("Reference", reference.image)]
    crops = [("Input", raw.image.crop((x, y, x+width, y+height))),
             ("Reference", reference.image.crop((x, y, x+width, y+height)))]
    for variant in variants:
        if variant == "original":
            correction, prediction_raw = residual, unclamped
        elif variant == "spatial_mean":
            correction = np.broadcast_to(mean.astype(np.float32), residual.shape).copy()
            prediction_raw = image + correction
        elif variant.startswith("sigma"):
            sigma = int(variant.removeprefix("sigma"))
            smooth = gaussian_filter(centered, sigma=(sigma, sigma, 0), mode="reflect", truncate=4.0)
            correction = (smooth + (mean - smooth.mean(axis=(0, 1), dtype=np.float64))).astype(np.float32)
            prediction_raw = image + correction
        elif correction_transform is not None:
            correction = correction_transform(variant, image, residual)
            if correction.shape != residual.shape or not np.isfinite(correction).all():
                raise ValueError("Counterfactual transform returned invalid residuals.")
            prediction_raw = image + correction
        else:
            raise ValueError(f"Unknown counterfactual variant: {variant}")
        mean_error = np.abs(correction.mean(axis=(0, 1), dtype=np.float64) - mean).max()
        if mean_error > 2e-7:
            raise ValueError("Ablation changed the mean residual instead of just spatial variation.")
        prediction = prediction_raw.clip(0, 1)
        with torch.inference_mode():
            prediction_tensor = tensor(prediction)
            score_psnr = psnr(prediction_tensor, reference_tensor).item()
            score_ssim = ssim(prediction_tensor, reference_tensor).item()
        if variant == "original" and (abs(score_psnr-float(cached_record["psnr"])) > 1e-4 or abs(score_ssim-cached_record["ssim"]) > 1e-5):
            raise ValueError("Ablation baseline metrics differ from cached native evaluation.")
        png = Image.fromarray(np.rint(prediction*255).astype(np.uint8))
        exported = export_png(png, output / f"{variant}.png")
        with Image.open(exported) as reopened:
            if reopened.size != raw.image.size or reopened.tobytes() != png.tobytes():
                raise ValueError("Ablation export changed its pixels or native geometry.")
        corrected_roi = correction[y:y+height, x:x+width]
        record = {"variant": variant, "psnr": score_psnr, "ssim": score_ssim,
                  "strictly_clipped_channel_fraction": float(((prediction_raw < 0) | (prediction_raw > 1)).mean()),
                  "mean_residual_rgb": correction.mean((0, 1), dtype=np.float64).tolist(),
                  "maximum_mean_residual_error": float(mean_error),
                  "residual_spatial_std_rgb": correction.std((0, 1), dtype=np.float64).tolist(),
                  "contour_roi_residual_std_rgb": corrected_roi.std((0, 1), dtype=np.float64).tolist(),
                  "residual_gradient_mean_abs": float((np.abs(np.diff(correction, axis=0)).mean()+np.abs(np.diff(correction, axis=1)).mean())/2),
                  "png_sha256": file_hash(exported), "elapsed_seconds": time.perf_counter()-started}
        records.append(record)
        previews.append((variant, png))
        crops.append((variant, png.crop((x, y, x+width, y+height))))
        print(identifier, variant, json.dumps({"psnr": score_psnr, "ssim": score_ssim, "clipped": record["strictly_clipped_channel_fraction"]}), flush=True)
        del prediction_tensor

    overview = Image.new("RGB", (6*280, 280), "white")
    detail = Image.new("RGB", (6*width, height+30), "white")
    for i, ((title, full), (_, crop)) in enumerate(zip(previews, crops)):
        thumb = ImageOps.contain(full, (276, 225))
        overview.paste(thumb, (i*280, 48))
        ImageDraw.Draw(overview).text((i*280+3, 5), f"{identifier} {title}\nDiagnostic counterfactual", fill="black")
        detail.paste(crop, (i*width, 30))
        ImageDraw.Draw(detail).text((i*width+3, 5), title, fill="black")
    overview.save(output / "overview.jpg", quality=94)
    detail.save(output / "native_contour.png")
    report = {"id": identifier, "split": "diagnostic", "model_forwards": 0, "training_steps": 0,
              "scope": "Offline counterfactual ablation, not a trained candidate or approved enhancement.",
              "held_fixed": ["input/reference", "original checkpoint", "native geometry", "global mean residual", "output clipping", "metric protocol"],
              "weights_sha256": provenance["weights_sha256"], "manifest_sha256": file_hash(manifest_path),
              "source_sha256": raw.source_sha256, "reference_sha256": reference.source_sha256,
              "cached_arrays_sha256": file_hash(npz), "metric_version": METRIC_VERSION,
              "smoothing": method_metadata or "Gaussian sigma 8/32 native pixels, reflect boundaries, truncate=4; spatial dimensions only; mean restored before clipping.",
              "contour_roi_xywh": roi, "records": records, "wall_seconds": time.perf_counter()-started,
              "selection_images_used": 0, "test_images_used": 0, "challenging_images_used": 0}
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", choices=["380_img_", "729_img_"], action="append")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/phase2/diagnosis_v2/residual_ablation")
    args = parser.parse_args()
    torch.set_num_threads(2)
    for identifier in args.id or ["380_img_", "729_img_"]:
        result = process(identifier, args.output)
        print(json.dumps({"id": identifier, "status": "complete", "seconds": result["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
