"""Verify cached V3 selection evidence/packets without model evaluation."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from aurora.image import validate_image
from aurora.manifest import file_hash, read_manifest, safe_path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    evaluation = ROOT / 'artifacts/phase2/evaluation_v3'
    ledger = json.loads((evaluation / 'ledger.json').read_text())
    config_path = ROOT / 'configs/rgb_baseline_v3.json'
    config = json.loads(config_path.read_text())
    manifest_path = ROOT / config['manifest']
    manifest = read_manifest(manifest_path)
    require(file_hash(manifest_path) == config['manifest_sha256'] == ledger['manifest_sha256'], 'Manifest identity')
    rows = {r['id']: r for r in manifest['pairs'] if r['split'] == 'selection'}
    require(len(rows) == 40 and len(ledger['candidates']) == 3, 'Declared scope')
    source_pixels = {}
    source_dimensions = {}
    for identifier, row in rows.items():
        for key in ['raw', 'reference']:
            valid = validate_image(safe_path(ROOT / config['data_root'], row[key + '_path']))
            require(valid.source_sha256 == row[key + '_sha256'], 'Frozen selection source')
            source_pixels[identifier, key] = hashlib.sha256(valid.image.tobytes()).hexdigest()
            source_dimensions[identifier, key] = valid.image.size
    findings, packet_hashes = [], {}
    for candidate in ledger['candidates']:
        bundle_path = Path(candidate['bundle'])
        bundle = json.loads(bundle_path.read_text())
        require(file_hash(bundle_path) == candidate['bundle_sha256'], 'Bundle identity')
        require(bundle['configuration'] == config and bundle['provenance']['config_sha256'] == file_hash(config_path), 'Candidate protocol')
        require(file_hash(bundle_path.parent / bundle['weights']) == bundle['weights_sha256'] == candidate['weights_sha256'], 'Weights identity')
        folder = evaluation / candidate['name'] / 'selection'
        summary_path = folder / 'summary.json'
        require(file_hash(summary_path) == candidate['selection_summary_sha256'], 'Selection summary identity')
        summary = json.loads(summary_path.read_text())
        identity = json.loads((folder / 'identity.json').read_text())
        require(identity['split'] == 'selection' and identity['bundle_sha256'] == candidate['bundle_sha256']
                and identity['weights_sha256'] == candidate['weights_sha256'] and identity['manifest_sha256'] == config['manifest_sha256']
                and identity['model'] == config['model'] and identity['scene_context'] == config['scene_context'], 'Evaluation identity')
        records = summary['records']
        require(summary['complete'] and summary['count'] == len(records) == 40 and {r['id'] for r in records} == set(rows), 'Complete selection membership')
        packet = evaluation / 'review' / candidate['label']
        require(len(list(packet.glob('overview-*.png'))) == 10 and len(list((packet / 'images').glob('*.png'))) == 120, 'Masked packet coverage')
        for record in records:
            identifier = record['id']; row = rows[identifier]
            wrapper = json.loads((folder / 'records' / (identifier + '.json')).read_text())
            require(wrapper['record'] == record and hashlib.sha256(json.dumps(record, sort_keys=True, allow_nan=False).encode()).hexdigest() == wrapper['record_sha256'], 'Metric record identity')
            require(record['raw_sha256'] == row['raw_sha256'] and record['reference_sha256'] == row['reference_sha256'], 'Record sources')
            image_path = folder / 'images' / (identifier + '.png')
            require(file_hash(image_path) == record['png_sha256'], 'Prediction PNG identity')
            with Image.open(image_path) as im:
                require(im.mode == 'RGB' and im.size == source_dimensions[identifier, 'raw'], 'Native output geometry')
                predicted_pixels = hashlib.sha256(im.tobytes()).hexdigest()
            for suffix, expected in [('original', source_pixels[identifier, 'raw']),
                                     ('candidate', predicted_pixels), ('reference', source_pixels[identifier, 'reference'])]:
                path = packet / 'images' / f'{identifier}-{suffix}.png'
                with Image.open(path) as image:
                    require(image.mode == 'RGB' and image.size == source_dimensions[identifier, 'raw']
                            and hashlib.sha256(image.tobytes()).hexdigest() == expected, 'Masked native triplet identity')
                packet_hashes[str(path.relative_to(ROOT))] = file_hash(path)
            require(all(np.isfinite(float(record[k])) for k in ['psnr', 'ssim', 'input_psnr', 'input_ssim']), 'Finite scores')
        means = {k: float(np.mean([float(r[k]) for r in records])) for k in ['psnr', 'ssim', 'input_psnr', 'input_ssim']}
        for metric in ['psnr', 'ssim']:
            require(np.isclose(means[metric], candidate['mean_' + metric], atol=1e-12, rtol=0)
                    and np.isclose(means[metric], summary['mean_' + metric], atol=1e-12, rtol=0), 'Mean arithmetic')
        for path in list(packet.glob('overview-*.png')) + [packet / 'README.txt']:
            packet_hashes[str(path.relative_to(ROOT))] = file_hash(path)
        findings.append({'name': candidate['name'], 'label': candidate['label'], 'means': means,
                         'count': 40, 'summary_sha256': file_hash(summary_path)})
    require(all(r['means']['input_psnr'] == findings[0]['means']['input_psnr'] and r['means']['input_ssim'] == findings[0]['means']['input_ssim'] for r in findings), 'Shared input baseline')
    report = {'status': 'selection_evidence_verified_reviews_pending', 'ledger_sha256': file_hash(evaluation / 'ledger.json'),
              'manifest_sha256': config['manifest_sha256'], 'config_sha256': file_hash(config_path),
              'candidates': findings, 'review_packet_hashes': packet_hashes,
              'review_script_sha256': file_hash(Path(__file__)),
              'selection_evaluations_verified': 120, 'masked_native_images_verified': 360,
              'model_forwards': 0, 'training_steps': 0, 'metrics_recomputed': False,
              'test_images_used': 0, 'challenging_images_used': 0, 'selected': False, 'phase2_complete': False}
    out = ROOT / 'artifacts/phase2/diagnosis_v3/selection_verification'; out.mkdir(exist_ok=False)
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'review_packet_hashes'}, indent=2))


if __name__ == '__main__':
    main()
