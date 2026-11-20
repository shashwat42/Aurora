"""Input-guided diagnostic residual constraint; no network inference or training.

Scalar-guide local linear guided filter (He, Sun, Tang, ECCV 2010),
with reflect boundary statistics and explicit global residual mean restoration.
Guide is mean encoded input RGB, never reference or predicted output.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter

import ablate_cached_residuals as baseline


def guided_residual(guide, residual, radius, epsilon):
    if guide.shape != residual.shape[:2] or residual.ndim != 3 or residual.shape[2] != 3:
        raise ValueError("Guide and RGB residual dimensions differ.")
    if type(radius) is not int or radius < 1 or not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Positive integer radius and finite positive epsilon are required.")
    if not np.isfinite(guide).all() or not np.isfinite(residual).all():
        raise ValueError("Guide/residual values must be finite.")
    g = guide.astype(np.float64)
    size = 2*radius+1

    def mean(array):
        return uniform_filter(array, size=size, mode="reflect")

    mean_g = mean(g)
    variance = np.maximum(mean(g*g)-mean_g*mean_g, 0)
    output = np.empty_like(residual, dtype=np.float32)
    for channel in range(3):
        p = residual[..., channel].astype(np.float64)
        mean_p = mean(p)
        covariance = mean(g*p)-mean_g*mean_p
        a = covariance/(variance+epsilon)
        b = mean_p-a*mean_g
        filtered = mean(a)*g+mean(b)
        filtered += p.mean()-filtered.mean()
        output[..., channel] = filtered.astype(np.float32)
    return output


def self_checks():
    # Mathematical invariants plus an edge-preservation example, no model/image data.
    guide = np.zeros((31, 31), dtype=np.float64)
    guide[:, 15:] = 1
    residual = np.repeat((.2*guide)[..., None], 3, axis=2).astype(np.float32)
    guided = guided_residual(guide, residual, 4, 1e-4)
    flat = guided_residual(np.zeros_like(guide), residual, 4, 1e-4)
    guided_jump = float((guided[:, 15]-guided[:, 14]).mean())
    flat_jump = float((flat[:, 15]-flat[:, 14]).mean())
    if guided_jump < .19 or flat_jump > .03:
        raise AssertionError("Guide edge was not preserved relative to matched flat-guide control.")
    constant = np.empty_like(residual)
    constant[...] = [.1, -.03, .2]
    for g in [guide, np.zeros_like(guide)]:
        if not np.allclose(guided_residual(g, constant, 4, 1e-4), constant, atol=1e-7, rtol=0):
            raise AssertionError("Constant residual was not preserved.")
    mean_error = float(np.abs(guided.mean((0, 1), dtype=np.float64)-residual.mean((0, 1), dtype=np.float64)).max())
    if mean_error > 2e-7:
        raise AssertionError("Global mean residual drifted.")
    # For a constant guide, the definition reduces to two local means.
    expected = np.stack([uniform_filter(uniform_filter(residual[..., c].astype(np.float64), size=9, mode="reflect"), size=9, mode="reflect") for c in range(3)], axis=-1)
    expected += residual.mean((0, 1), dtype=np.float64)-expected.mean((0, 1))
    if not np.allclose(flat, expected, atol=1e-7, rtol=0):
        raise AssertionError("Flat-guide control does not match its independent box-filter identity.")
    return {"status": "passed", "constant_residual_preserved": True, "flat_guide_identity": True,
            "synthetic_guided_step_jump": guided_jump, "synthetic_flat_step_jump": flat_jump,
            "maximum_mean_error": mean_error}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=baseline.ROOT / "configs/diagnostic_edge_aware_pilot.json")
    parser.add_argument("--id", choices=["380_img_", "729_img_"], action="append")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    checks = self_checks()
    if args.self_test:
        print(json.dumps(checks, indent=2))
        return
    config = json.loads(args.config.read_text())
    output = args.output or baseline.ROOT / config["output"]
    output.mkdir(parents=True, exist_ok=True)
    recorded = output / "protocol.json"
    if recorded.exists() and json.loads(recorded.read_text()) != config:
        raise ValueError("Output already belongs to another diagnostic protocol.")
    baseline.write_json(recorded, config)
    baseline.write_json(output / "self_checks.json", checks)

    def transform(variant, image, residual):
        settings = config[variant]
        guide = image.mean(axis=2, dtype=np.float64)
        if variant == "flat32":
            guide = np.zeros_like(guide)
        return guided_residual(guide, residual, settings["radius"], settings["epsilon"])

    baseline.torch.set_num_threads(2)
    for identifier in args.id or config["diagnostic_ids"]:
        result = baseline.process(identifier, output, variants=config["variants"],
            correction_transform=transform,
            method_metadata={"method": "Scalar input-guided local-linear residual filtering", "protocol": config,
                "note": "Flat32 matches guided32's window/statistics but has no input structure; all controls retain the global residual mean before clipping."})
        print(json.dumps({"id": identifier, "status": "complete", "seconds": result["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
