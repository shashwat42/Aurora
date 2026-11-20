"""Fixed-eight, reference-assisted data-fit bounds; no inference or training.

Transforms use each image's reference and are deliberately NOT deployable models.
Run from Code with .venv/bin/python scripts/reassess_rgb_fit.py.
"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from aurora.image import validate_image
from aurora.manifest import file_hash, read_manifest, safe_path
from PIL import Image


def components(prediction, reference):
    error = prediction - reference
    bias = error.mean(axis=(0, 1))
    mse = float(np.mean(error ** 2))
    mean_mse = float(np.mean(bias ** 2))
    centered = float(np.mean((error - bias) ** 2))
    if not np.isclose(mse, mean_mse + centered, atol=1e-12):
        raise ValueError('MSE decomposition failed')
    return {'mse': mse, 'mean_bias_mse': mean_mse, 'centered_mse': centered,
            'mean_bias_fraction': mean_mse / mse if mse else 0.0}


def affine_oracle(rgb, reference):
    """Per-channel least squares BEFORE clipping; no claim of clipped optimum."""
    x = rgb.reshape(-1, 3)
    y = reference.reshape(-1, 3)
    xc, yc = x - x.mean(0), y - y.mean(0)
    variance = np.mean(xc ** 2, axis=0)
    gain = np.divide(np.mean(xc * yc, axis=0), variance,
                     out=np.zeros(3), where=variance > 1e-15)
    offset = y.mean(0) - gain * x.mean(0)
    return np.clip(rgb * gain + offset, 0, 1), gain, offset


def residual_floor(rgb, reference, scale=0.5):
    """Infimum MSE for ANY independent per-pixel residual bounded by scale.

    More permissive than a shared finite network; tanh endpoints are limits.
    """
    excess = np.maximum(np.abs(reference - rgb) - scale, 0)
    return {'unreachable_channel_fraction': float(np.mean(excess > 0)),
            'mse_infimum': float(np.mean(excess ** 2))}


def main():
    protocol_path = ROOT / 'configs/valid_center_pilot.json'
    protocol = json.loads(protocol_path.read_text())
    spec = protocol['reuse_diagnostic_evaluation']['control']
    summary_path = ROOT / spec['summary']
    if file_hash(summary_path) != spec['summary_sha256']:
        raise ValueError('Control summary changed')
    summary = json.loads(summary_path.read_text())
    records = {r['id']: r for r in summary['records']}
    manifest_path = ROOT / 'data/metadata/manifest.v1.json'
    rows = {r['id']: r for r in read_manifest(manifest_path)['pairs']
            if r['split'] == 'diagnostic'}
    ids = protocol['native_edge_ids']
    if len(ids) != 8 or len(set(ids)) != 8 or not set(ids) <= rows.keys():
        raise ValueError('Expected the eight pre-existing diagnostic cases')
    findings = []
    for identifier in ids:
        row = rows[identifier]
        arrays, hashes = [], {}
        for key in ['raw', 'reference']:
            valid = validate_image(safe_path(ROOT / 'data', row[key + '_path']))
            if valid.source_sha256 != row[key + '_sha256']:
                raise ValueError('Frozen source changed')
            hashes[key] = valid.source_sha256
            arrays.append(np.asarray(valid.image, dtype=np.float64) / 255)
        path = summary_path.parent / 'images' / (identifier + '.png')
        hashes['control_png'] = file_hash(path)
        if hashes['control_png'] != records[identifier]['png_sha256']:
            raise ValueError('Control export changed')
        with Image.open(path) as image:
            control = np.asarray(image.convert('RGB'), dtype=np.float64) / 255
        raw, ref = arrays
        if raw.shape != ref.shape or raw.shape != control.shape:
            raise ValueError('Geometry mismatch')
        affine, gain, offset = affine_oracle(raw, ref)
        # This corrects only the spatial mean of the existing prediction.
        mean_adjusted = np.clip(control + (ref - control).mean((0, 1)), 0, 1)
        findings.append({'id': identifier, 'sha256': hashes,
                         'input': components(raw, ref), 'control': components(control, ref),
                         'reference_fitted_input_affine': components(affine, ref),
                         'reference_fitted_control_mean': components(mean_adjusted, ref),
                         'affine_gain': gain.tolist(), 'affine_offset': offset.tolist(),
                         'residual_limit': residual_floor(raw, ref)})
    report = {
        'scope': 'Eight adaptively inspected diagnostic cases; no generalization claim',
        'method': 'Native decoded RGB in [0,1]; cached 8-bit control; equal image weighting',
        'reference_assisted': True,
        'warning': 'Oracle transforms use the target. Not deployable candidates, held-out results, or evidence of learnability.',
        'manifest_sha256': file_hash(manifest_path),
        'case_protocol_sha256': file_hash(protocol_path),
        'control_summary_sha256': file_hash(summary_path),
        'script_sha256': file_hash(Path(__file__)),
        'model_forwards': 0, 'training_steps': 0, 'selection_images_used': 0,
        'test_images_used': 0, 'challenging_images_used': 0,
        'selected': False, 'phase2_complete': False, 'findings': findings,
        'means': {key: float(np.mean([r[key]['mse'] for r in findings]))
                  for key in ['input', 'control', 'reference_fitted_input_affine',
                              'reference_fitted_control_mean']},
        'residual_mse_infimum_mean': float(np.mean([r['residual_limit']['mse_infimum'] for r in findings]))}
    out = ROOT / 'artifacts/phase2/diagnosis_v2/baseline_reassessment'
    out.mkdir(exist_ok=False)
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'means': report['means'], 'cases': [
        {'id': r['id'], 'bias_fraction': r['control']['mean_bias_fraction'],
         'affine_vs_control_mse': r['reference_fitted_input_affine']['mse'] / r['control']['mse'],
         'mean_adjusted_vs_control_mse': r['reference_fitted_control_mean']['mse'] / r['control']['mse'],
         'residual_limit': r['residual_limit']} for r in findings]}, indent=2))


if __name__ == '__main__':
    main()
