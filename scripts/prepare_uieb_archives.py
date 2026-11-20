"""Extract original UIEB archives safely and inventory them without creating splits.
Run from the project root: .venv/bin/python scripts/prepare_uieb_archives.py
Requires rarfile, UnRAR, Pillow, numpy. Existing images must match archive contents.
"""
from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import json
import os
import rarfile

rarfile.UNRAR_TOOL = os.environ.get("UNRAR_TOOL", "unrar")
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1] / "data"
UIEB = ROOT / 'uieb'

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    archives = []
    for archive_name, category, password in [('raw-890.rar', 'raw', '1234567'), ('reference-890.rar', 'reference', '8901234'), ('challenging-60.rar', 'challenging', '5678901')]:
        archive = UIEB / 'archives' / archive_name
        archives.append({'path': str(archive.relative_to(ROOT)), 'bytes': archive.stat().st_size, 'sha256': sha256(archive)})
        with rarfile.RarFile(str(archive)) as entries:
            for entry in entries.infolist():
                if entry.isdir():
                    continue
                p = Path(entry.filename)
                if p.is_absolute() or '..' in p.parts or not entry.is_file():
                    raise ValueError(f'Unsafe archive entry: {entry.filename}')
                if p.suffix.lower() not in {'.png', '.jpg', '.jpeg'}:
                    raise ValueError(f'Unexpected file: {entry.filename}')
                dest = UIEB / category / p.name
                content = entries.read(entry, pwd=password)
                if dest.exists():
                    if dest.read_bytes() != content:
                        raise ValueError(f'Existing image differs: {dest}')
                else:
                    dest.write_bytes(content)
        print(f'Extracted {archive_name}', flush=True)
    rows, errors, warnings = [], [], []
    duplicates = {}
    by_category = {}
    for category in ['raw', 'reference', 'challenging']:
        records, pixel_hashes = {}, defaultdict(list)
        for path in sorted((UIEB / category).iterdir()):
            if not path.is_file():
                continue
            record = {'id': path.stem, 'category': category, 'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size, 'sha256': sha256(path)}
            try:
                with Image.open(path) as im:
                    record.update(width=im.width, height=im.height, mode=im.mode, format=im.format, frames=getattr(im, 'n_frames', 1))
                    im.load()
                    rgb = im.convert('RGB')
                    h = hashlib.sha256(f'{rgb.size}:'.encode() + np.asarray(rgb).tobytes()).hexdigest()
                    record['rgb_sha256'] = h
                    pixel_hashes[h].append(path.name)
                    if im.format not in ['PNG', 'JPEG'] or record['frames'] != 1:
                        warnings.append({'path': record['path'], 'reason': 'Nonstandard encoding or multiple frames; preserve original and review before training.'})
            except Exception as exc:
                errors.append({'path': record['path'], 'error': str(exc)})
            if path.stem in records:
                raise ValueError(f'Duplicate identifier: {category}/{path.stem}')
            records[path.stem] = record
            rows.append(record)
        by_category[category] = records
        duplicates[category] = [v for v in pixel_hashes.values() if len(v) > 1]
    raw, reference = by_category['raw'], by_category['reference']
    missing_reference = sorted(raw.keys() - reference.keys())
    missing_raw = sorted(reference.keys() - raw.keys())
    pairs = []
    for key in sorted(raw.keys() & reference.keys()):
        a, b = raw[key], reference[key]
        same_size = (a.get('width'), a.get('height')) == (b.get('width'), b.get('height'))
        pairs.append({'id': key, 'raw_path': a['path'], 'reference_path': b['path'], 'width': a.get('width'), 'height': a.get('height'), 'dimensions_match': same_size, 'split': 'unassigned'})
    metadata = ROOT / 'metadata'
    metadata.mkdir(exist_ok=True)
    for name, records in [('files.csv', rows), ('uieb_pairs.csv', pairs)]:
        fields = list(dict.fromkeys(k for r in records for k in r))
        with (metadata / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    audit = {'dataset': 'UIEB', 'checked_on': '2026-11-01', 'counts': {k: len(v) for k, v in by_category.items()}, 'paired_ids': len(pairs), 'missing_reference': missing_reference, 'missing_raw': missing_raw, 'dimension_mismatches': [r['id'] for r in pairs if not r['dimensions_match']], 'decode_errors': errors, 'warnings': warnings, 'identical_rgb_groups': duplicates, 'archives': archives, 'split_status': 'Unassigned; duplicate, related-scene and visual alignment review required before freezing splits.', 'note': 'Duplicate detection uses decoded RGB pixels; equal dimensions do not prove visual alignment. Original file bytes are preserved.'}
    (metadata / 'uieb_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit, indent=2))
    if errors or missing_reference or missing_raw or audit['dimension_mismatches']:
        raise SystemExit('Audit found pair or decoding errors; see metadata/uieb_audit.json')

if __name__ == '__main__':
    main()
