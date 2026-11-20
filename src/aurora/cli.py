import argparse
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(prog="aurora")
    commands = parser.add_subparsers(dest="command", required=True)
    app = commands.add_parser("app", help="Launch the local RGB enhancement prototype")
    app.add_argument("--port", type=int, default=7860)
    app.add_argument("--bundle", help="Explicit experimental RGB model bundle")
    app.add_argument("--preview-only", action="store_true")
    check = commands.add_parser("validate", help="Validate an RGB JPEG/PNG")
    check.add_argument("image")
    train = commands.add_parser("train", help="Train the Phase 2 RGB reconstruction model")
    train.add_argument("--config", default="configs/rgb_baseline.json")
    pilot = commands.add_parser("identity-pilot", help="Run the declared two-arm diagnostic-only identity-preservation experiment")
    pilot.add_argument("--config", default="configs/identity_pilot.json")
    context = commands.add_parser("context-pilot", help="Run the declared valid-center reconstruction diagnostic pilot")
    context.add_argument("--config", default="configs/valid_center_pilot.json")
    infer = commands.add_parser("infer", help="Run a trained RGB bundle and export native-size PNG")
    infer.add_argument("image")
    infer.add_argument("--bundle", required=True)
    infer.add_argument("--output", required=True)
    infer.add_argument("--threads", type=int, default=4)
    evaluate = commands.add_parser("evaluate-candidates", help="Evaluate all declared RGB candidates at native size; generate masked review packets")
    evaluate.add_argument("--run", default="artifacts/phase2/rgb_baseline")
    evaluate.add_argument("--output", default="artifacts/phase2/evaluation")
    evaluate.add_argument("--threads", type=int, default=4)
    evaluate.add_argument("--diagnostic-only", action="store_true", help="Evaluate diagnostic images only before considering selection; no review packets or checkpoint selection")
    select = commands.add_parser("select-baseline", help="Apply the artifact gate and TRN03 after two complete independent reviews")
    select.add_argument("--evaluation", default="artifacts/phase2/evaluation")
    for name in ["depth-preflight", "cache-depth"]:
        p = commands.add_parser(name)
        p.add_argument("--config", default="configs/depth.json")
        p.add_argument("--manifest", default="data/metadata/manifest.v1.json")
        p.add_argument("--data-root", default="data")
        p.add_argument("--limit", type=int)
        p.add_argument("--allow-draft", action="store_true", help="Preparation-only diagnostic/cache run on provisional memberships; does not permit training")
        if name == "cache-depth":
            p.add_argument("--split", choices=["train", "diagnostic", "selection"], default="train")
    args = parser.parse_args()
    if args.command == "app":
        from .app import launch
        launch(args.port, args.bundle, args.preview_only)
    elif args.command == "validate":
        from .image import validate_image
        valid = validate_image(args.image)
        print(json.dumps({"width": valid.image.width, "height": valid.image.height, "source_sha256": valid.source_sha256}))
    elif args.command == "train":
        print('Loading PyTorch training dependencies; imports may take a moment...', flush=True)
        from .training.run import train
        print(json.dumps(train(args.config), indent=2))
    elif args.command == "identity-pilot":
        from .training.pilot import run_identity_pilot
        print(json.dumps(run_identity_pilot(args.config), indent=2))
    elif args.command == "context-pilot":
        from .training.pilot import run_context_pilot
        print(json.dumps(run_context_pilot(args.config), indent=2))
    elif args.command == "evaluate-candidates":
        print('Loading evaluation dependencies; imports may take a moment...', flush=True)
        from .evaluation.rgb import evaluate_candidates
        print(json.dumps(evaluate_candidates(args.run, args.output, threads=args.threads,
                                             diagnostic_only=args.diagnostic_only), indent=2))
    elif args.command == "select-baseline":
        from .evaluation.rgb import finalize_selection
        print(json.dumps(finalize_selection(args.evaluation), indent=2))
    elif args.command == "infer":
        import torch
        from .image import validate_image
        from .inference import RGBInference, export_png
        from .manifest import read_manifest
        if args.threads < 1:
            parser.error("--threads must be positive")
        torch.set_num_threads(args.threads)
        valid = validate_image(args.image)
        model = RGBInference(args.bundle)
        # Known final inputs stay sealed during the development workflow.
        manifest = read_manifest(model.bundle["configuration"]["manifest"])
        sealed = [r for r in manifest["pairs"] if r["split"] == "test"] + manifest.get("challenging", [])
        if valid.source_sha256 in {r["raw_sha256"] for r in sealed}:
            raise ValueError("Final test and challenging inputs remain sealed during development.")
        image, report = model.enhance(valid)
        export_png(image, args.output)
        print(json.dumps({**report, "output": str(args.output)}, indent=2))
    else:
        from .depth import DepthEstimator
        from .image import validate_image
        from .manifest import read_manifest, development_rows, safe_path, file_hash
        model = DepthEstimator(args.config)
        split = "diagnostic" if args.command == "depth-preflight" else args.split
        manifest = read_manifest(args.manifest, require_frozen=not args.allow_draft)
        rows = development_rows(manifest, split)
        if args.limit is not None:
            if args.limit < 1:
                parser.error("--limit must be positive")
            rows = rows[:args.limit]
        out = Path("artifacts/depth_preflight" if args.command == "depth-preflight" else "artifacts/cache_jobs")
        out.mkdir(parents=True, exist_ok=True)
        evidence = []
        start = time.perf_counter()
        for row in rows:
            valid = validate_image(safe_path(args.data_root, row["raw_path"]))
            if valid.source_sha256 != row["raw_sha256"]:
                raise ValueError("Input changed after manifest freeze.")
            depth, meta = model.cache(valid)
            evidence.append({"id": row["id"], **meta})
            if args.command == "depth-preflight":
                from .diagnostics import save_depth_review
                save_depth_review(valid.image, depth, out / row["id"])
            report = {"split": split, "complete": len(evidence) == len(rows), "processed": len(evidence),
                      "requested": len(rows), "wall_seconds": time.perf_counter() - start,
                      "manifest_status": manifest["status"], "manifest_sha256": file_hash(args.manifest),
                      "estimator_fingerprint": model.fingerprint, "records": evidence}
            # A partial progress record survives interruptions; never labels a partial cache complete.
            report_path = out / f"{split}.json"
            temp = report_path.with_suffix(".json.tmp")
            temp.write_text(json.dumps(report, indent=2) + "\n")
            temp.replace(report_path)
            print(f"{len(evidence)}/{len(rows)} {row['id']} {meta['seconds']:.3f}s", flush=True)


if __name__ == "__main__":
    main()
