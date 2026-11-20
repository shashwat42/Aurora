"""Post-selection evidence audit, not rerating, tuning or model evaluation."""
import csv
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from aurora.manifest import file_hash


def main():
    evaluation = ROOT / 'artifacts/phase2/evaluation_v3'
    packet = evaluation / 'review/candidate-C'
    selection = json.loads((evaluation / 'selection.json').read_text())
    audit = json.loads((ROOT / 'artifacts/phase2/diagnosis_v3/artifact_review_verification/report.json').read_text())
    reviews = []
    for number in [1, 2]:
        path = packet / f'reviewer-{number}.csv'
        if file_hash(path) != audit['review_csv_sha256'][str(path.relative_to(ROOT))]:
            raise ValueError('Recorded ratings changed')
        with path.open(newline='') as stream:
            reviews.append({r['id']: r for r in csv.DictReader(stream)})
    p1_path = ROOT / 'artifacts/phase2/agent_reviews_v3/reviewer1/provenance.json'
    p2_path = ROOT / 'artifacts/phase2/agent_reviews_v3/reviewer2/provenance.json'
    context = selection['review_context']
    for path in [p1_path, p2_path]:
        if file_hash(path) != context['provenance_files'][str(path.relative_to(ROOT))]:
            raise ValueError('Review provenance changed')
    p1, p2 = [json.loads(p.read_text()) for p in [p1_path, p2_path]]
    disagreements = [i for i in reviews[0] if reviews[0][i]['severity'] != reviews[1][i]['severity']]
    if len(disagreements) != 28:
        raise ValueError('Expected 28 recorded epoch20 disagreements')
    boundary_ids = {i for i in disagreements if ('Thin colored line visible along actual outer image boundary' in reviews[1][i]['notes']
                                                or 'faint colored outer edge' in reviews[1][i]['notes'])}
    magnified = {r['id']: r['box'] for r in p2['magnified_boundary_check']}
    targeted = {r['id']: r['box'] for r in p2['additional_native_views']}
    center = {r['id']: r['crop'] for r in p2['inputs']}
    snapshot = json.loads((ROOT / 'artifacts/phase2/diagnosis_v3/selection_verification/report.json').read_text())
    out = ROOT / 'artifacts/phase2/diagnosis_v3/disagreement_audit'; out.mkdir(exist_ok=False)
    findings = []
    native_rows, edge_rows = [], []
    for identifier in disagreements:
        images = []
        hashes = {}
        for suffix in ['original', 'candidate', 'reference']:
            path = packet / 'images' / f'{identifier}-{suffix}.png'
            digest = file_hash(path)
            if digest != snapshot['review_packet_hashes'][str(path.relative_to(ROOT))]:
                raise ValueError('Review image changed')
            hashes[suffix] = digest
            with Image.open(path) as image:
                images.append(image.convert('RGB'))
        w, h = images[0].size
        category = 'outer_boundary' if identifier in boundary_ids else 'colour_highlight_or_texture'
        box = magnified.get(identifier)
        if identifier in boundary_ids and box is None:
            box = [max(0, w - 64), 0, w, min(64, h)]
        if box:
            # An abrupt outermost-to-adjacent difference is descriptive only;
            # real scene edges can also produce it. No severity threshold.
            x0, y0, x1, y1 = box
            edge = 'right' if x1 == w else 'left' if x0 == 0 else 'top'
            measures = {}
            for suffix, image in zip(['original', 'candidate', 'reference'], images):
                a = np.asarray(image, dtype=np.float64)
                if edge == 'right':
                    outer, inner, next_inner = a[y0:y1, -1], a[y0:y1, -2], a[y0:y1, -3]
                elif edge == 'left':
                    outer, inner, next_inner = a[y0:y1, 0], a[y0:y1, 1], a[y0:y1, 2]
                else:
                    outer, inner, next_inner = a[0, x0:x1], a[1, x0:x1], a[2, x0:x1]
                measures[suffix] = {'outer_adjacent_mean_abs_8bit': float(np.mean(np.abs(outer-inner))),
                                    'adjacent_next_mean_abs_8bit': float(np.mean(np.abs(inner-next_inner))),
                                    'signed_outer_adjacent_mean_rgb_8bit': (outer-inner).mean(0).tolist()}
            row = Image.new('RGB', (840, 220), 'white'); draw = ImageDraw.Draw(row)
            for column, (suffix, image) in enumerate(zip(['input', 'output', 'reference'], images)):
                crop = image.crop(box)
                draw.text((column * 280 + 3, 3), f'{identifier} {suffix}: native / 3x', fill='black')
                row.paste(crop, (column * 280 + 3, 24))
                row.paste(crop.resize((crop.width * 3, crop.height * 3), Image.Resampling.NEAREST), (column * 280 + 80, 24))
            edge_rows.append(row)
        else:
            measures = None
        if identifier not in boundary_ids:
            region = targeted.get(identifier, center[identifier])
            x0, y0, x1, y1 = region
            # Same-coordinate bounded native sample; no rescaling.
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            cw, ch = min(320, w), min(240, h)
            x0 = min(max(0, cx-cw//2), w-cw); y0 = min(max(0, cy-ch//2), h-ch)
            native_box = [x0, y0, x0+cw, y0+ch]
            row = Image.new('RGB', (990, 270), 'white'); draw = ImageDraw.Draw(row)
            for column, (suffix, image) in enumerate(zip(['input', 'output', 'reference'], images)):
                draw.text((column*330+3, 3), f'{identifier} {suffix} native', fill='black')
                row.paste(image.crop(native_box), (column*330+3, 25))
            native_rows.append(row)
        else:
            native_box = None
        findings.append({'id': identifier, 'category_from_recorded_notes': category,
                         'reviewer1': reviews[0][identifier], 'reviewer2': reviews[1][identifier],
                         'source_png_sha256': hashes, 'boundary_box': box, 'boundary_measures': measures,
                         'other_native_box': native_box})
    for name, rows in [('boundary', edge_rows), ('other_native', native_rows)]:
        for start in range(0, len(rows), 4):
            batch = rows[start:start+4]
            sheet = Image.new('RGB', (batch[0].width, batch[0].height * len(batch)), 'white')
            for index, row in enumerate(batch):sheet.paste(row, (0, index*row.height))
            sheet.save(out / f'{name}-{start//4+1}.png')
    report = {'status': 'evidence_prepared_visual_interpretation_pending',
              'scope': 'Post-selection audit of 28 recorded epoch20 disagreements, not new independent ratings or tuning',
              'selection_sha256': file_hash(evaluation / 'selection.json'),
              'rating_csv_sha256': {str(packet / f'reviewer-{i}.csv'): file_hash(packet / f'reviewer-{i}.csv') for i in [1,2]},
              'disagreement_count': 28, 'boundary_note_count': len(boundary_ids),
              'other_note_count': len(disagreements)-len(boundary_ids),
              'boundary_measure_units': 'Decoded 8-bit RGB levels in recorded corner strips. No artifact severity threshold; true image edges may be abrupt.',
              'reviewer1_general_native_coverage': 'Upper-left, center, lower-right plus targeted subjects',
              'reviewer2_general_native_coverage': 'Center, upper-right, lower-left plus targeted subjects and magnified boundary checks',
              'findings': findings, 'model_forwards': 0, 'training_steps': 0,
              'metrics_recomputed': False, 'ratings_changed': False, 'decision_changed': False,
              'diagnostic_images_used': 0, 'test_images_used': 0, 'challenging_images_used': 0,
              'selection_images_read_for_evidence_audit': 28,
              'script_sha256': file_hash(Path(__file__))}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['findings','rating_csv_sha256']}, indent=2))


if __name__ == '__main__':main()
