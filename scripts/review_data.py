"""Create a reproducible integrity/similarity review, before any split assignment."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.fft import dctn
from aurora.image import validate_image

OUT = ROOT / "artifacts/data_review"
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data"
rows = list(csv.DictReader((DATA / "metadata/uieb_pairs.csv").open()))


def analyze(row):
    a = validate_image(DATA / row["raw_path"])
    b = validate_image(DATA / row["reference_path"])
    if a.image.size != b.image.size:
        raise ValueError(f"Oriented dimensions differ: {row['id']}")
    thumb = a.image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    pixels = np.asarray(thumb, dtype=np.float32)
    spectrum = dctn(pixels, norm="ortho")[:8, :8].ravel()[1:]
    phash = spectrum > np.median(spectrum)
    centered = pixels.ravel() - pixels.mean()
    centered /= max(float(np.linalg.norm(centered)), 1e-8)
    # A translation diagnostic only, not a proof of scene alignment.
    x = np.asarray(a.image.convert("L").resize((128, 128)), dtype=np.float32)
    y = np.asarray(b.image.convert("L").resize((128, 128)), dtype=np.float32)
    fx, fy = np.fft.fft2(x - x.mean()), np.fft.fft2(y - y.mean())
    cross = fx * fy.conj()
    corr = np.fft.ifft2(cross / np.maximum(np.abs(cross), 1e-8)).real
    shift = np.array(np.unravel_index(corr.argmax(), corr.shape))
    shift[shift > 64] -= 128
    return {**row, "raw_sha256": a.source_sha256, "reference_sha256": b.source_sha256,
            "width": a.image.width, "height": a.image.height,
            "shift_y_128": int(shift[0]), "shift_x_128": int(shift[1]),
            "below_256": min(a.image.size) < 256}, phash, centered


with ThreadPoolExecutor(max_workers=4) as pool:
    analyzed = list(pool.map(analyze, rows))
records = [x[0] for x in analyzed]
hashes = np.array([x[1] for x in analyzed])
features = np.array([x[2] for x in analyzed])
similarities = features @ features.T
candidates = []
for i in range(len(rows)):
    distances = np.count_nonzero(hashes[i + 1:] != hashes[i], axis=1)
    for offset in np.where((distances <= 10) | (similarities[i, i + 1:] >= 0.94))[0]:
        j = i + 1 + int(offset)
        candidates.append({"a": rows[i]["id"], "b": rows[j]["id"], "phash_hamming": int(distances[offset]),
                           "gray_correlation": float(similarities[i, j])})
review = {"method": "63-bit perceptual DCT hash <=10 OR normalized 32x32 grayscale correlation >=0.94; manually review candidates, conservatively group uncertain relations. Not a guarantee of exhaustive scene discovery.",
          "pairs": records, "candidates": candidates,
          "alignment_flags": [r["id"] for r in records if max(abs(r["shift_x_128"]), abs(r["shift_y_128"])) > 1],
          "below_256": [r["id"] for r in records if r["below_256"]]}
(OUT / "review.json").write_text(json.dumps(review, indent=2) + "\n")


def tile(path, size):
    with Image.open(path) as im:
        return ImageOps.contain(im.convert("RGB"), size)


# All paired inputs/references, 50 pairs per sheet; no holdout is assigned yet.
for start in range(0, len(rows), 50):
    sheet = Image.new("RGB", (1800, 1250), "#20252b")
    draw = ImageDraw.Draw(sheet)
    for k, row in enumerate(rows[start:start + 50]):
        x, y = (k % 5) * 360, (k // 5) * 125
        draw.text((x + 4, y + 2), row["id"] + "   input | reference", fill="white")
        for column, name in enumerate(["raw_path", "reference_path"]):
            im = tile(DATA / row[name], (176, 103))
            sheet.paste(im, (x + column * 180, y + 20))
    sheet.save(OUT / f"all_pairs_{start // 50 + 1:02d}.jpg", quality=90)

by_id = {r["id"]: r for r in rows}
for start in range(0, len(candidates), 12):
    sheet = Image.new("RGB", (1200, 1152), "#20252b")
    draw = ImageDraw.Draw(sheet)
    for k, pair in enumerate(candidates[start:start + 12]):
        x, y = (k % 3) * 400, (k // 3) * 288
        draw.text((x + 4, y + 2), f"{pair['a']} / {pair['b']}  hash={pair['phash_hamming']}", fill="white")
        for column, key in enumerate(["a", "b"]):
            im = tile(DATA / by_id[pair[key]]["raw_path"], (195, 255))
            sheet.paste(im, (x + column * 200, y + 22))
    sheet.save(OUT / f"candidates_{start // 12 + 1:02d}.jpg", quality=92)
print(json.dumps({"pairs": len(rows), "candidates": len(candidates), "alignment_flags": review["alignment_flags"], "below_256": review["below_256"]}, indent=2))
