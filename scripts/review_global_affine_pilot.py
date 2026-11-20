"""Verify cached global-affine pilot identities/aggregates without evaluation."""
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from aurora.manifest import file_hash, read_manifest, safe_path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    config_path = ROOT / 'configs/global_affine_pilot.json'
    config = json.loads(config_path.read_text())
    run = ROOT / config['output']
    summary = json.loads((run / 'summary.json').read_text())
    model = json.loads((run / 'model.json').read_text())
    manifest_path = ROOT / config['manifest']
    require(file_hash(manifest_path) == config['manifest_sha256'] == model['manifest_sha256'], 'Manifest identity')
    manifest = read_manifest(manifest_path)
    train_ids = [r['id'] for r in manifest['pairs'] if r['split'] == 'train']
    diagnostic = {r['id']: r for r in manifest['pairs'] if r['split'] == 'diagnostic'}
    require(len(train_ids) == 713 and model['train_ids'] == train_ids, 'Training membership')
    require(json.loads((run / 'protocol.json').read_text()) == config, 'Recorded protocol')
    require(file_hash(config_path) == model['protocol_sha256'] == summary['protocol_sha256'], 'Protocol identity')
    require(file_hash(run / 'model.json') == summary['model_sha256'], 'Model identity')
    require(file_hash(ROOT / 'scripts/run_global_affine_pilot.py') == model['script_sha256'], 'Pilot source identity')
    require(file_hash(ROOT / 'scripts/reassess_rgb_fit.py') == model['fit_helper_sha256'], 'Fit helper identity')
    control_path = ROOT / config['control_summary']
    require(file_hash(control_path) == config['control_summary_sha256'] == summary['control_summary_sha256'], 'Control identity')
    control = {r['id']: r for r in json.loads(control_path.read_text())['records']}
    records = summary['records']
    require(len(records) == 40 and {r['id'] for r in records} == set(diagnostic) == set(control), 'Diagnostic membership')
    deltas = []
    for record in records:
        identifier = record['id']; row = diagnostic[identifier]
        for key in ['raw', 'reference']:
            require(file_hash(safe_path(ROOT / config['data_root'], row[key + '_path'])) == row[key + '_sha256'] == record[key + '_sha256'], 'Source identity')
        path = run / 'images' / (identifier + '.png')
        require(file_hash(path) == record['png_sha256'], 'Output identity')
        with Image.open(path) as output, Image.open(safe_path(ROOT / config['data_root'], row['raw_path'])) as raw:
            require(output.mode == 'RGB' and output.size == raw.size, 'Native output geometry')
        require(len(record['coefficients']) == 6 and np.isfinite(record['coefficients']).all(), 'Finite coefficients')
        require(0 <= record['clipped_channel_fraction'] <= 1, 'Clipping fraction')
        for metric in ['psnr', 'ssim']:
            require(np.isfinite(record[metric]), 'Finite metric')
            require(record['control_' + metric] == control[identifier][metric], 'Control metric')
        deltas.append({'id': identifier, **{m + '_delta': record[m] - record['control_' + m] for m in ['psnr', 'ssim']}})
    means = {m: float(np.mean([r[m] for r in records])) for m in ['psnr', 'ssim', 'control_psnr', 'control_ssim']}
    require(means == summary['means'], 'Aggregate metrics')
    numeric_pass = all(means[m] >= means['control_' + m] for m in ['psnr', 'ssim'])
    require(numeric_pass == summary['numeric_continuation_pass'], 'Continuation decision')
    require(not numeric_pass and not summary['selected'] and not summary['phase2_complete'], 'Expected failed pilot')
    require(summary['train_count'] == 713 and summary['diagnostic_count'] == 40 and summary['model_reload_exact'], 'Run completion')
    require(all(summary[k + '_images_used'] == 0 for k in ['selection', 'test', 'challenging']), 'Held-out counts')
    report = {'status': 'verified_failed_not_promoted', 'summary_sha256': file_hash(run / 'summary.json'),
              'review_script_sha256': file_hash(Path(__file__)), 'means': means,
              'mean_deltas': {m: means[m] - means['control_' + m] for m in ['psnr', 'ssim']},
              'wins': {m: sum(r[m + '_delta'] > 0 for r in deltas) for m in ['psnr', 'ssim']},
              'both_wins': sum(r['psnr_delta'] > 0 and r['ssim_delta'] > 0 for r in deltas),
              'paired_deltas': deltas, 'numeric_continuation_pass': False,
              'model_forwards': 0, 'training_steps': 0, 'metrics_recomputed': False,
              'selected': False, 'phase2_complete': False,
              'decision': 'Close this fixed regression without feature/alpha sweeps; no new baseline approved'}
    out = ROOT / 'artifacts/phase2/diagnosis_v2/global_affine_review'
    out.mkdir(exist_ok=False)
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'paired_deltas'}, indent=2))


if __name__ == '__main__':
    main()
