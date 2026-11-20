"""Audit completed V3 records and render fixed-eight cached comparisons."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from aurora.manifest import file_hash, read_manifest, safe_path
from aurora.image import validate_image


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    config_path = ROOT / 'configs/rgb_baseline_v3.json'
    config = json.loads(config_path.read_text())
    run, evaluation = ROOT / config['output'], ROOT / 'artifacts/phase2/evaluation_v3'
    status = json.loads((run / 'status.json').read_text())
    provenance = json.loads((run / 'provenance.json').read_text())
    ledger = json.loads((evaluation / 'diagnostic_ledger.json').read_text())
    manifest_path = ROOT / config['manifest']
    manifest = read_manifest(manifest_path)
    require(file_hash(manifest_path) == config['manifest_sha256'] == provenance['manifest_sha256'] == ledger['manifest_sha256'], 'Manifest identity')
    require(json.loads((run / 'config.json').read_text()) == config and file_hash(config_path) == provenance['config_sha256'], 'Config identity')
    require(all(file_hash(ROOT / 'src/aurora' / name) == digest for name, digest in provenance['source_sha256'].items()), 'Source identity')
    require(file_hash(ROOT / 'requirements.lock.txt') == provenance['lock_sha256'], 'Dependency lock')
    require(status['status'] == 'complete' and status['steps'] == 7140 and status['epochs_reached'] == 20
            and status['finite_losses'] and status['checkpoint_reload_exact'], 'Training completion')
    train_ids = {r['id'] for r in manifest['pairs'] if r['split'] == 'train'}
    rows = {r['id']: r for r in manifest['pairs'] if r['split'] == 'diagnostic'}
    require(len(train_ids) == 713 and len(rows) == 40, 'Partition sizes')
    counts, epoch_steps = {}, Counter()
    with (run / 'steps.jsonl').open() as stream:
        for index, line in enumerate(stream, 1):
            record = json.loads(line); epoch = record['epoch']
            require(record['step'] == index and epoch == (index - 1) // 357 + 1, 'Step/epoch sequence')
            require(all(np.isfinite(record[k]) for k in ['loss', 'l1', 'ssim_loss', 'gradient_norm']), 'Finite training record')
            require(len(record['ids']) == (1 if index % 357 == 0 else 2), 'Batch size')
            counts.setdefault(epoch, Counter()).update(record['ids']); epoch_steps[epoch] += 1
    require(index == 7140 and set(counts) == set(range(1, 21)), 'Training log coverage')
    require(all(c == Counter({i: 1 for i in train_ids}) for c in counts.values()), 'Every train member once per epoch')
    expected = [f'epoch-{e:03d}' for e in config['candidate_epochs']]
    require(status['candidate_bundles'] == expected == [c['name'] for c in ledger['candidates']], 'Candidate budget')
    require(ledger['status'] == 'diagnostic_only_complete' and not ledger['selected'] and ledger['selection_images_used'] == 0, 'Diagnostic scope')
    control_path = ROOT / 'artifacts/phase2/identity_pilot/evaluation/control/summary.json'
    prior_protocol = json.loads((ROOT / 'configs/valid_center_pilot.json').read_text())
    require(file_hash(control_path) == prior_protocol['reuse_diagnostic_evaluation']['control']['summary_sha256'], 'Control identity')
    control = json.loads(control_path.read_text())
    control_records = {r['id']: r for r in control['records']}
    require(set(control_records) == set(rows), 'Control membership')
    verified, summaries = [], {}
    for candidate in ledger['candidates']:
        name = candidate['name']; directory = evaluation / name / 'diagnostic'
        bundle_path = run / name / 'bundle.json'
        bundle = json.loads(bundle_path.read_text())
        require(bundle['configuration'] == config and bundle['provenance'] == provenance, 'Bundle protocol')
        require(bundle['architecture'] == config['architecture'] and bundle['model'] == config['model'], 'Bundle architecture')
        require(file_hash(bundle_path.parent / 'weights.pt') == candidate['weights_sha256'] == bundle['weights_sha256'], 'Weights identity')
        require(file_hash(bundle_path.parent / 'training_state.pt') == bundle['training_state_sha256'], 'Optimizer identity')
        require(bundle['training_steps'] == candidate['training_steps'] == bundle['epoch'] * 357, 'Candidate steps')
        summary = json.loads((directory / 'summary.json').read_text()); summaries[name] = summary
        identity = json.loads((directory / 'identity.json').read_text())
        require(identity['weights_sha256'] == bundle['weights_sha256'] and identity['bundle_sha256'] == file_hash(bundle_path)
                and identity['model'] == config['model'] and identity['scene_context'] == config['scene_context']
                and identity['split'] == 'diagnostic' and identity['manifest_sha256'] == config['manifest_sha256'], 'Evaluation identity')
        require(all(summary[k] == v for k, v in identity.items()), 'Summary identity')
        records = summary['records']
        require(summary['complete'] and summary['count'] == len(records) == 40 and {r['id'] for r in records} == set(rows), 'Evaluation coverage')
        for record in records:
            identifier = record['id']; row = rows[identifier]
            wrapper = json.loads((directory / 'records' / (identifier + '.json')).read_text())
            require(wrapper['record'] == record and hashlib.sha256(json.dumps(record, sort_keys=True, allow_nan=False).encode()).hexdigest() == wrapper['record_sha256'], 'Metric record checksum')
            require(file_hash(directory / 'images' / (identifier + '.png')) == record['png_sha256'], 'PNG identity')
            require(record['raw_sha256'] == row['raw_sha256'] and record['reference_sha256'] == row['reference_sha256'], 'Source record identity')
            require(all(np.isfinite(float(record[m])) for m in ['psnr', 'ssim', 'input_psnr', 'input_ssim']), 'Finite metrics')
        for metric in ['psnr', 'ssim']:
            mean = float(np.mean([float(r[metric]) for r in records]))
            require(np.isclose(mean, summary['mean_' + metric], atol=1e-12) and np.isclose(mean, candidate['diagnostic_mean_' + metric], atol=1e-12), 'Aggregate metrics')
        deltas = {m: summary['mean_' + m] - control['mean_' + m] for m in ['psnr', 'ssim']}
        verified.append({'name': name, 'weights_sha256': bundle['weights_sha256'],
                         'summary_sha256': file_hash(directory / 'summary.json'),
                         'mean_psnr': summary['mean_psnr'], 'mean_ssim': summary['mean_ssim'],
                         'delta_vs_control': deltas, 'numeric_continuation_pass': all(v >= 0 for v in deltas.values())})
    last = json.loads((run / 'last/bundle.json').read_text())
    require(file_hash(run / 'last/weights.pt') == last['weights_sha256'] == status['weights_sha256'] == verified[-1]['weights_sha256'], 'Last weights identity')
    require(not (evaluation / 'ledger.json').exists() and not (evaluation / 'selection.json').exists()
            and not any(evaluation.glob('epoch-*/selection')), 'Selection remains unopened')
    # Verify bytes only, not metrics/model predictions, on the diagnostic inputs.
    for row in rows.values():
        for key in ['raw', 'reference']:
            require(file_hash(safe_path(ROOT / config['data_root'], row[key + '_path'])) == row[key + '_sha256'], 'Frozen diagnostic bytes')
    out = ROOT / 'artifacts/phase2/diagnosis_v3/review'; out.mkdir(exist_ok=False)
    ids = prior_protocol['native_edge_ids']
    rois = {r['id']: r['native_crop_xywh'] for r in json.loads((ROOT / 'artifacts/phase2/diagnosis_v2/identity_regressions/report.json').read_text())['findings']}
    sources = {'Control': control_path.parent, **{name: evaluation / name / 'diagnostic' for name in expected}}
    indexes = {'Control': control_records, **{name: {r['id']: r for r in summaries[name]['records']} for name in expected}}
    visual_records = []
    for offset in [0, 4]:
        overview = Image.new('RGB', (6 * 280, 4 * 240), 'white'); od = ImageDraw.Draw(overview)
        crops = Image.new('RGB', (6 * 272, 4 * 285), 'white'); cd = ImageDraw.Draw(crops)
        corners = Image.new('RGB', (6 * 144, 4 * 160), 'white'); ed = ImageDraw.Draw(corners)
        for index, identifier in enumerate(ids[offset:offset + 4]):
            row = rows[identifier]
            raw = validate_image(safe_path(ROOT / config['data_root'], row['raw_path'])).image
            ref = validate_image(safe_path(ROOT / config['data_root'], row['reference_path'])).image
            images = [raw]; hashes = {}
            for name, directory in sources.items():
                path = directory / 'images' / (identifier + '.png')
                hashes[name] = file_hash(path)
                require(hashes[name] == indexes[name][identifier]['png_sha256'], 'Visual source identity')
                with Image.open(path) as image:
                    images.append(image.convert('RGB'))
            images.append(ref)
            require(all(im.size == raw.size for im in images), 'Visual native dimensions')
            x, y, w, h = rois[identifier]
            for col, (name, image) in enumerate(zip(['Input', *sources, 'Reference'], images)):
                label = identifier + ' ' + name
                overview.paste(ImageOps.contain(image, (280, 210)), (col * 280, index * 240 + 25)); od.text((col * 280 + 2, index * 240 + 3), label, fill='black')
                crops.paste(image.crop((x, y, x + w, y + h)), (col * 272, index * 285 + 25)); cd.text((col * 272 + 2, index * 285 + 3), label, fill='black')
                corners.paste(image.crop((0, 0, min(128, image.width), min(128, image.height))), (col * 144, index * 160 + 25)); ed.text((col * 144 + 2, index * 160 + 3), label, fill='black')
            visual_records.append({'id': identifier, 'native_crop_xywh': rois[identifier], 'png_sha256': hashes})
        for name, image in [('overview', overview), ('native_crops', crops), ('native_corners', corners)]:
            image.save(out / f'{name}-{offset // 4 + 1}.png')
    report = {'status': 'numeric_verified_native_review_pending', 'config_sha256': file_hash(config_path),
              'ledger_sha256': file_hash(evaluation / 'diagnostic_ledger.json'),
              'control_summary_sha256': file_hash(control_path), 'training_steps_verified': 7140,
              'epochs_verified': 20, 'training_images_per_epoch_verified': 713,
              'candidates': verified, 'visual_cases': visual_records,
              'review_script_sha256': file_hash(Path(__file__)),
              'model_forwards': 0, 'training_steps_run': 0, 'full_metrics_recomputed': False,
              'selection_images_used': 0, 'test_images_used': 0, 'challenging_images_used': 0,
              'selected': False, 'phase2_complete': False}
    (out / 'verification.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'visual_cases'}, indent=2))


if __name__ == '__main__':
    main()
