"""Local review images; depth is a relative cue, not measured distance."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps


def save_depth_review(rgb, depth, prefix):
    prefix = Path(prefix)
    gray = Image.fromarray(np.rint(depth * 255).astype(np.uint8))
    gray.save(prefix.with_suffix(".depth.png"))
    edges = np.asarray(gray.filter(ImageFilter.FIND_EDGES), dtype=np.float32).copy()
    edges[:2] = edges[-2:] = 0
    edges[:, :2] = edges[:, -2:] = 0
    mask = edges > max(16, np.percentile(edges, 97))
    overlay = np.asarray(rgb).copy()
    overlay[mask] = [255, 60, 50]
    Image.fromarray(overlay).save(prefix.with_suffix(".overlay.png"))
    sheet = Image.new("RGB", (1500, 530), "#20252b")
    draw = ImageDraw.Draw(sheet)
    for i, (im, label) in enumerate([(rgb, "Oriented input"), (gray.convert("RGB"), "Relative depth (near = bright)"), (Image.fromarray(overlay), "Depth boundaries over RGB")]):
        draw.text((i * 500 + 5, 5), label, fill="white")
        sheet.paste(ImageOps.contain(im, (495, 495)), (i * 500, 30))
    sheet.save(prefix.with_suffix(".review.jpg"), quality=92)
