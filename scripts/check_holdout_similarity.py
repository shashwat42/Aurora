"""Flag paired/challenging scene candidates for data curation, never model tuning."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.fft import dctn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/data_review"
files = list(csv.DictReader((ROOT / "data/metadata/files.csv").open()))
raw = [r for r in files if r["category"] == "raw"]
challenging = [r for r in files if r["category"] == "challenging" and r["format"] in {"PNG", "JPEG"} and r["frames"] == "1"]
def fingerprint(row):
    with Image.open(ROOT / "data" / row["path"]) as image:
        a = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    spectrum = dctn(a, norm="ortho")[:8, :8].ravel()[1:]
    a = a.ravel() - a.mean()
    return spectrum > np.median(spectrum), a / max(float(np.linalg.norm(a)), 1e-8)
with ThreadPoolExecutor(max_workers=4) as pool:
    raw_features = list(pool.map(fingerprint, raw))
    challenge_features = list(pool.map(fingerprint, challenging))
candidates = []
for i, row in enumerate(challenging):
    for j, other in enumerate(raw):
        distance = int(np.count_nonzero(challenge_features[i][0] != raw_features[j][0]))
        corr = float(challenge_features[i][1] @ raw_features[j][1])
        if distance <= 12 or corr >= 0.94:
            candidates.append({"paired_id": other["id"], "challenging_id": row["id"], "phash_hamming": distance,
                               "gray_correlation": corr, "raw_path": other["path"], "challenging_path": row["path"]})
report = {"paired_inputs_checked": len(raw), "supported_challenging_checked": len(challenging),
          "candidates": candidates, "method": "63-bit DCT pHash <=12 OR grayscale correlation >=0.94", "status": "requires visual/source-scene review; absence of candidates is not proof of independence"}
(OUT / "holdout_similarity.json").write_text(json.dumps(report, indent=2) + "\n")
for start in range(0, len(candidates), 8):
    sheet = Image.new("RGB", (1000, 1000), "#20252b")
    draw = ImageDraw.Draw(sheet)
    for k, c in enumerate(candidates[start:start + 8]):
        x, y = k % 2 * 500, k // 2 * 250
        draw.text((x + 3, y + 3), f"paired {c['paired_id']} | challenging {c['challenging_id']}", fill="white")
        for col, key in enumerate(["raw_path", "challenging_path"]):
            with Image.open(ROOT / "data" / c[key]) as image:
                sheet.paste(ImageOps.contain(image.convert("RGB"), (245, 220)), (x + col * 250, y + 25))
    sheet.save(OUT / f"holdout_candidates_{start // 8 + 1:02d}.jpg", quality=92)
print(json.dumps(report, indent=2))
