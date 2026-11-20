"""User-run train-only colour-fit pilot; never a release/selection bundle."""
import json
from pathlib import Path
import sys

if __name__ == '__main__':
    print('Loading NumPy, Pillow and PyTorch; dependency imports may take a moment...', flush=True)

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from aurora.image import validate_image
from aurora.manifest import file_hash, read_manifest, safe_path
from aurora.training.metrics import psnr, ssim
from reassess_rgb_fit import affine_oracle


def features(rgb):
    pixels = rgb.reshape(-1, 3)
    return np.concatenate([pixels.mean(0), pixels.std(0),
                           np.quantile(pixels, [0.1, 0.5, 0.9], axis=0).ravel()])


def fit_ridge(x, y, alpha):
    if alpha <= 0 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Finite training data and positive alpha required')
    center, scale = x.mean(0), x.std(0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    z, intercept = (x - center) / scale, y.mean(0)
    weights = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]),
                              z.T @ (y - intercept))
    return {'center': center.tolist(), 'scale': scale.tolist(),
            'intercept': intercept.tolist(), 'weights': weights.tolist()}


def predict_coefficients(model, rgb):
    z = (features(rgb) - np.asarray(model['center'])) / np.asarray(model['scale'])
    result = z @ np.asarray(model['weights']) + np.asarray(model['intercept'])
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite predicted coefficients')
    return result


def source(row, key, data_root):
    valid = validate_image(safe_path(data_root, row[key + '_path']))
    if valid.source_sha256 != row[key + '_sha256']:
        raise ValueError('Frozen source changed')
    return np.asarray(valid.image, dtype=np.float64) / 255


def main(config_path=None):
    print('Dependencies loaded. Checking frozen data and control identities...', flush=True)
    config_path = Path(config_path) if config_path else ROOT / 'configs/global_affine_pilot.json'
    config = json.loads(config_path.read_text())
    manifest_path = ROOT / config['manifest']
    if file_hash(manifest_path) != config['manifest_sha256']:
        raise ValueError('Wrong manifest')
    manifest = read_manifest(manifest_path)
    train = [r for r in manifest['pairs'] if r['split'] == 'train']
    diagnostic = [r for r in manifest['pairs'] if r['split'] == 'diagnostic']
    if len(train) != config['expected_train_count'] or len(diagnostic) != config['expected_diagnostic_count']:
        raise ValueError('Unexpected partition counts')
    control_path = ROOT / config['control_summary']
    if file_hash(control_path) != config['control_summary_sha256']:
        raise ValueError('Control summary changed')
    control = json.loads(control_path.read_text())
    controls = {r['id']: r for r in control['records']}
    if len(controls) != len(diagnostic) or set(controls) != {r['id'] for r in diagnostic}:
        raise ValueError('Control diagnostic membership mismatch')
    out = ROOT / config['output']
    out.mkdir(exist_ok=False)
    (out / 'protocol.json').write_text(json.dumps(config, indent=2) + '\n')
    data_root = ROOT / config['data_root']
    print(f'Extracting training features from {len(train)} pairs...', flush=True)
    x, y = [], []
    for index, row in enumerate(train):
        raw, ref = source(row, 'raw', data_root), source(row, 'reference', data_root)
        if raw.shape != ref.shape:
            raise ValueError('Training geometry mismatch')
        _, gain, offset = affine_oracle(raw, ref)
        x.append(features(raw)); y.append(np.concatenate([gain, offset]))
        if index == 0 or (index + 1) % 50 == 0 or index + 1 == len(train):
            print(f'Training features {index + 1}/{len(train)}', flush=True)
    print('Fitting the fixed ridge regression...', flush=True)
    model = fit_ridge(np.asarray(x), np.asarray(y), config['ridge_alpha'])
    model.update({'run_kind': 'diagnostic_pilot', 'selected': False,
                  'train_ids': [r['id'] for r in train],
                  'manifest_sha256': file_hash(manifest_path),
                  'protocol_sha256': file_hash(config_path),
                  'script_sha256': file_hash(Path(__file__)),
                  'fit_helper_sha256': file_hash(ROOT / 'scripts/reassess_rgb_fit.py')})
    (out / 'model.json').write_text(json.dumps(model, indent=2, allow_nan=False) + '\n')
    restored = json.loads((out / 'model.json').read_text())
    if model != restored:
        raise ValueError('Model JSON reload mismatch')
    model = restored
    torch.set_num_threads(config['threads'])
    (out / 'images').mkdir()
    results = []
    print(f'Evaluating {len(diagnostic)} diagnostic images at native size...', flush=True)
    for index, row in enumerate(diagnostic):
        raw = source(row, 'raw', data_root)
        # Predict before loading diagnostic reference; coefficients use input only.
        coefficients = predict_coefficients(model, raw)
        unbounded = raw * coefficients[:3] + coefficients[3:]
        prediction = np.clip(unbounded, 0, 1).astype(np.float32)
        ref = source(row, 'reference', data_root).astype(np.float32)
        if ref.shape != prediction.shape:
            raise ValueError('Diagnostic geometry mismatch')
        tensors = [torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1))).unsqueeze(0)
                   for a in [prediction, ref]]
        with torch.inference_mode():
            metrics = {'psnr': float(psnr(*tensors).item()),
                       'ssim': float(ssim(*tensors).item())}
        image_path = out / 'images' / (row['id'] + '.png')
        Image.fromarray(np.rint(prediction * 255).astype(np.uint8)).save(image_path)
        results.append({'id': row['id'], **metrics, 'coefficients': coefficients.tolist(),
                        'clipped_channel_fraction': float(np.mean((unbounded < 0) | (unbounded > 1))),
                        'png_sha256': file_hash(image_path),
                        'raw_sha256': row['raw_sha256'], 'reference_sha256': row['reference_sha256'],
                        'control_psnr': controls[row['id']]['psnr'],
                        'control_ssim': controls[row['id']]['ssim']})
        print(f'Diagnostic {index + 1}/{len(diagnostic)}: {row["id"]}', flush=True)
    means = {m: float(np.mean([r[m] for r in results])) for m in ['psnr', 'ssim', 'control_psnr', 'control_ssim']}
    numeric_pass = all(means[m] >= means['control_' + m] for m in ['psnr', 'ssim'])
    summary = {'status': 'diagnostic_complete', 'protocol_sha256': file_hash(config_path),
               'model_sha256': file_hash(out / 'model.json'),
               'control_summary_sha256': file_hash(control_path),
               'metric_version': 'rgb-psnr-ssim-valid11-v1',
               'train_count': len(train), 'diagnostic_count': len(results),
               'model_reload_exact': True,
               'means': means, 'numeric_continuation_pass': numeric_pass,
               'decision': 'Native fixed-eight visual review required before any continuation' if numeric_pass else 'Do not promote or automatically sweep features/alpha',
               'selected': False, 'phase2_complete': False,
               'selection_images_used': 0, 'test_images_used': 0, 'challenging_images_used': 0,
               'records': results}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'records'}, indent=2))
    return summary


if __name__ == '__main__':
    main()
