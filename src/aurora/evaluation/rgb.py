import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch

from ..image import validate_image
from ..inference import RGBInference, export_png
from ..manifest import development_rows, file_hash, read_manifest, safe_path
from ..training.metrics import METRIC_VERSION, psnr, ssim
from ..training.run import write_json


def json_score(value):
    value = float(value)
    if math.isnan(value):
        raise ValueError("A metric produced NaN.")
    return value if math.isfinite(value) else "Infinity"


def evaluate_bundle(bundle_path, split, output, *, threads=4):
    """Resumable, checksummed evaluation; final partitions are never accepted."""
    if split not in {"diagnostic", "selection"}:
        raise ValueError("Only diagnostic and selection evaluation is permitted.")
    if threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(threads)
    model = RGBInference(bundle_path)
    kind = model.bundle["run_kind"]
    if kind == "diagnostic_pilot" and split != "diagnostic":
        raise ValueError("Diagnostic pilot bundles cannot enter selection evaluation.")
    if kind not in {"baseline_candidate", "diagnostic_pilot"}:
        raise ValueError("A training smoke checkpoint cannot enter baseline selection.")
    config = model.bundle["configuration"]
    if split == 'selection' and config.get('diagnostic_gate_record'):
        gate_path = Path(config['diagnostic_gate_record'])
        if not gate_path.exists():
            raise ValueError('Diagnostic acceptance record required before selection evaluation.')
        gate = json.loads(gate_path.read_text())
        if (gate.get('status') != 'diagnostic_accepted_for_selection'
                or gate.get('config_sha256') != model.bundle['provenance']['config_sha256']
                or gate.get('manifest_sha256') != config['manifest_sha256']
                or model.bundle['weights_sha256'] not in gate.get('candidate_weights_sha256', [])
                or not gate.get('fixed_native_review_passed')):
            raise ValueError('Diagnostic acceptance does not match this candidate/protocol or lacks native review.')
    manifest = read_manifest(config["manifest"])
    digest = file_hash(config["manifest"])
    if digest != config["manifest_sha256"]:
        raise ValueError("Checkpoint manifest identity differs from the dataset.")
    rows = development_rows(manifest, split)
    if not rows:
        raise ValueError("Evaluation split is empty.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    identity = {"weights_sha256": model.bundle["weights_sha256"], "manifest_sha256": digest,
                "metric_version": METRIC_VERSION, "split": split, "protocol": "native-float-rgb-v1"}
    if config.get('scene_context'):
        # Moment policy and colour scales affect inference even with identical weights.
        identity.update({'architecture': model.bundle['architecture'], 'model': model.bundle['model'],
                         'scene_context': config['scene_context'], 'bundle_sha256': file_hash(bundle_path)})
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Existing evaluation belongs to different weights/data/metrics.")
    write_json(identity_path, identity)
    (output / "records").mkdir(exist_ok=True)
    (output / "images").mkdir(exist_ok=True)
    records = []
    for row in rows:
        record_path = safe_path(output / "records", row["id"] + ".json")
        image_path = safe_path(output / "images", row["id"] + ".png")
        valid = validate_image(safe_path(config["data_root"], row["raw_path"]))
        reference = validate_image(safe_path(config["data_root"], row["reference_path"]))
        if valid.source_sha256 != row["raw_sha256"] or reference.source_sha256 != row["reference_sha256"]:
            raise ValueError("Evaluation source bytes differ from the frozen manifest.")
        if valid.image.size != reference.image.size:
            raise ValueError("Native RGB/reference geometry differs.")
        if record_path.exists():
            cached = json.loads(record_path.read_text())
            record = cached["record"]
            encoded = json.dumps(record, sort_keys=True, allow_nan=False).encode()
            if hashlib.sha256(encoded).hexdigest() != cached["record_sha256"]:
                raise ValueError("Cached metric record checksum mismatch.")
            if record["raw_sha256"] != valid.source_sha256 or record["reference_sha256"] != reference.source_sha256 or file_hash(image_path) != record["png_sha256"]:
                raise ValueError("Cached evaluation was altered.")
        else:
            import time
            started = time.perf_counter()
            prediction = model.predict(valid)
            seconds = time.perf_counter() - started
            raw = torch.from_numpy(np.array(valid.image, dtype=np.float32).transpose(2, 0, 1).copy() / 255).unsqueeze(0)
            target = torch.from_numpy(np.array(reference.image, dtype=np.float32).transpose(2, 0, 1).copy() / 255).unsqueeze(0)
            with torch.inference_mode():
                scores = {"psnr": json_score(psnr(prediction, target).item()),
                          "ssim": ssim(prediction, target).item(),
                          "input_psnr": json_score(psnr(raw, target).item()),
                          "input_ssim": ssim(raw, target).item()}
            pixels = np.rint(prediction[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            # An interrupted export is replaced only inside this matching evaluation.
            image_path.unlink(missing_ok=True)
            export_png(Image.fromarray(pixels), image_path)
            record = {"id": row["id"], "scene_id": row["scene_id"], **scores,
                      "width": valid.image.width, "height": valid.image.height,
                      "raw_sha256": valid.source_sha256, "reference_sha256": reference.source_sha256,
                      "inference_seconds": seconds, "png_sha256": file_hash(image_path)}
            encoded = json.dumps(record, sort_keys=True, allow_nan=False).encode()
            write_json(record_path, {"record": record, "record_sha256": hashlib.sha256(encoded).hexdigest()})
        records.append(record)
        write_json(output / "progress.json", {**identity, "complete": len(records) == len(rows),
                    "processed": len(records), "requested": len(rows)})
        print(f"{Path(bundle_path).parent.name} {split} {len(records)}/{len(rows)} {row['id']}", flush=True)
    summary = {**identity, "complete": True, "count": len(records),
               "mean_psnr": json_score(statistics.mean(float(r["psnr"]) for r in records)),
               "mean_ssim": statistics.mean(r["ssim"] for r in records),
               "input_mean_psnr": json_score(statistics.mean(float(r["input_psnr"]) for r in records)),
               "input_mean_ssim": statistics.mean(r["input_ssim"] for r in records),
               "provisional_inference_p95_seconds": float(np.percentile([r["inference_seconds"] for r in records], 95)),
               "timing_scope": "CPU RGB tensor inference only; excludes export and worker startup. Not a Phase 3 performance contract.",
               "records": records}
    write_json(output / "summary.json", summary)
    return summary


def rank_candidates(candidates):
    """TRN03 ordering, called only on artifact-eligible candidates for selection."""
    if not candidates:
        raise ValueError("No eligible candidates.")
    highest = max(float(c["mean_psnr"]) for c in candidates)
    tied = [c for c in candidates if float(c["mean_psnr"]) == highest or highest - float(c["mean_psnr"]) <= .01 + 1e-12]
    return min(tied, key=lambda c: (-c["mean_ssim"], c["training_steps"]))


def _review_packet(output, ledger, config):
    """Masked packets: no checkpoint names, hashes, epoch numbers or scores."""
    review_root = output / "review"
    review_root.mkdir(exist_ok=True)
    rows = development_rows(read_manifest(config["manifest"]), "selection")
    for candidate in ledger["candidates"]:
        folder = review_root / candidate["label"]
        folder.mkdir(exist_ok=True)
        image_root = folder / "images"
        image_root.mkdir(exist_ok=True)
        thumbnails = []
        for row in rows:
            stem = row["id"]
            paths = [safe_path(config["data_root"], row["raw_path"]),
                     output / candidate["name"] / "selection/images" / (stem + ".png"),
                     safe_path(config["data_root"], row["reference_path"])]
            triptych = Image.new("RGB", (990, 280), "white")
            draw = ImageDraw.Draw(triptych)
            for i, (path, title, suffix) in enumerate(zip(paths, ["Original", "Candidate", "Reference"], ["original", "candidate", "reference"])):
                native_path = image_root / f"{stem}-{suffix}.png"
                if not native_path.exists():
                    # Original/reference use the same orientation/profile conversion.
                    clean = validate_image(path).image
                    export_png(clean, native_path)
                with Image.open(native_path) as image:
                    tile = ImageOps.contain(image, (320, 240))
                    triptych.paste(tile, (i * 330, 30))
                draw.text((i * 330 + 5, 5), f"{stem} — {title}", fill="black")
            thumbnails.append(triptych)
        for offset in range(0, len(thumbnails), 4):
            sheet = Image.new("RGB", (990, 280 * len(thumbnails[offset:offset+4])), "white")
            for i, tile in enumerate(thumbnails[offset:offset+4]):
                sheet.paste(tile, (0, i*280))
            sheet.save(folder / f"overview-{offset//4+1:02d}.png")
        # Templates are never overwritten after reviewers start filling them.
        for reviewer in [1, 2]:
            template = folder / f"reviewer-{reviewer}.csv"
            if not template.exists():
                with template.open("x", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["id", "reviewer", "independent", "severity", "notes"])
                    writer.writeheader()
                    writer.writerows({"id": r["id"], "reviewer": "", "independent": "", "severity": "", "notes": ""} for r in rows)
        (folder / "README.txt").write_text("Review every native-size original/candidate/reference triplet in images/. Overviews are navigation aids; inspect native boundaries, color, texture and noise. Do not consult scores or the private ledger before reviewing. Fill both CSVs independently with distinct reviewer identities; independent=yes, severity=none/minor/major, notes required for defects. Major: new/worsened false structure, lost content, strong halos, severe noise. Minor: other correction defects. No benefit or review approval is implied by these packets.\n")


def evaluate_candidates(run_path, output, *, threads=4, diagnostic_only=False):
    run_path, output = Path(run_path), Path(output)
    status = json.loads((run_path / "status.json").read_text())
    config = json.loads((run_path / "config.json").read_text())
    if status["status"] != "complete" or status["run_kind"] != "baseline_candidate":
        raise ValueError("The full baseline training run must complete first.")
    expected = [f"epoch-{e:03d}" for e in config["candidate_epochs"]]
    if status["candidate_bundles"] != expected or status["steps"] != math.ceil(len(development_rows(read_manifest(config["manifest"]), "train"))/config["batch_size"])*config["epochs"]:
        raise ValueError("Candidate set/training steps differ from the declared budget.")
    output.mkdir(parents=True, exist_ok=True)
    if diagnostic_only and (output / "ledger.json").exists():
        raise ValueError("This output already contains selection evaluation; choose a fresh diagnostic-only output directory.")
    labels = [f"candidate-{chr(65+i)}" for i in range(len(expected))]
    random.Random(config["seed"]).shuffle(labels)
    candidates = []
    for name, label in zip(expected, labels):
        bundle = run_path / name / "bundle.json"
        meta = json.loads(bundle.read_text())
        if meta["configuration"] != config or meta["epoch"] not in config["candidate_epochs"]:
            raise ValueError("Candidate configuration differs from the predeclared run.")
        diagnostic = evaluate_bundle(bundle, "diagnostic", output / name / "diagnostic", threads=threads)
        if diagnostic_only:
            candidates.append({"name": name, "bundle": str(bundle.resolve()),
                "weights_sha256": meta["weights_sha256"], "training_steps": meta["training_steps"],
                "diagnostic_mean_psnr": diagnostic["mean_psnr"], "diagnostic_mean_ssim": diagnostic["mean_ssim"]})
            continue
        selection = evaluate_bundle(bundle, "selection", output / name / "selection", threads=threads)
        candidates.append({"name": name, "label": label, "bundle": str(bundle.resolve()),
                           "bundle_sha256": file_hash(bundle), "weights_sha256": meta["weights_sha256"],
                           "training_steps": meta["training_steps"], "mean_psnr": selection["mean_psnr"],
                           "mean_ssim": selection["mean_ssim"], "diagnostic_mean_psnr": diagnostic["mean_psnr"],
                           "selection_summary_sha256": file_hash(output / name / "selection/summary.json")})
    if diagnostic_only:
        report = {"schema_version": 1, "status": "diagnostic_only_complete", "selected": False,
            "run": str(run_path.resolve()), "manifest_sha256": config["manifest_sha256"],
            "metric_version": METRIC_VERSION, "candidates": candidates, "selection_images_used": 0,
            "notice": "Development diagnostics only; no checkpoint selected and no selection/test/challenging images processed."}
        write_json(output / "diagnostic_ledger.json", report)
        return report
    ledger = {"schema_version": 1, "status": "awaiting_independent_artifact_reviews", "selected": False,
              "run": str(run_path.resolve()), "manifest_sha256": config["manifest_sha256"],
              "metric_version": METRIC_VERSION, "candidates": candidates,
              "provisional_metric_leader": rank_candidates(candidates)["name"],
              "notice": "Metric leader is for a clearly labeled prototype only. It is not an artifact-approved or selected baseline."}
    write_json(output / "ledger.json", ledger)
    _review_packet(output, ledger, config)
    return ledger


def finalize_selection(output):
    """Require two complete independent reviews per candidate; unresolved=max."""
    output = Path(output)
    ledger = json.loads((output / "ledger.json").read_text())
    eligible, screens, identities = [], {}, None
    for candidate in ledger["candidates"]:
        summary_path = output / candidate["name"] / "selection/summary.json"
        if file_hash(summary_path) != candidate["selection_summary_sha256"] or file_hash(candidate["bundle"]) != candidate["bundle_sha256"]:
            raise ValueError("Selection evidence was altered after evaluation.")
        summary = json.loads(summary_path.read_text())
        ids = {r["id"] for r in summary["records"]}
        reviews, names = [], []
        for index in [1, 2]:
            path = output / "review" / candidate["label"] / f"reviewer-{index}.csv"
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            if len(rows) != len(ids) or {r["id"] for r in rows} != ids:
                raise ValueError("Review membership is missing or duplicated.")
            people = {r["reviewer"].strip() for r in rows}
            if len(people) != 1 or not next(iter(people)):
                raise ValueError("Every review file needs one consistent named reviewer.")
            if any(r["independent"].strip().lower() != "yes" or r["severity"] not in {"none", "minor", "major"} or
                   (r["severity"] != "none" and not r["notes"].strip()) for r in rows):
                raise ValueError("Complete independent reviews and defect notes are required.")
            names.append(next(iter(people)))
            reviews.append({r["id"]: r["severity"] for r in rows})
        if len(set(n.casefold() for n in names)) != 2:
            raise ValueError("Two distinct independent reviewers are required.")
        if identities is None:
            identities = names
        elif names != identities:
            raise ValueError("Use the same two reviewers across candidates.")
        severity = {"none": 0, "minor": 1, "major": 2}
        highest = [max(severity[r[i]] for r in reviews) for i in ids]
        major, minor = highest.count(2), highest.count(1)
        passed = major == 0 and minor / len(ids) <= .1
        screens[candidate["name"]] = {"passed": passed, "major": major, "minor": minor,
                    "reviewers": names, "disagreement_policy": "unresolved cases use higher severity",
                    "review_sha256": [file_hash(output / "review" / candidate["label"] / f"reviewer-{i}.csv") for i in [1, 2]]}
        if passed:
            eligible.append(candidate)
    report = {"status": "selected" if eligible else "no_artifact_eligible_candidate", "selected": bool(eligible),
              "manifest_sha256": ledger["manifest_sha256"], "screens": screens,
              "chosen": rank_candidates(eligible) if eligible else None}
    context_path = output / "review_context.json"
    if context_path.exists():
        context = json.loads(context_path.read_text())
        if context.get("reviewer_ids") != identities:
            raise ValueError("Review context identities differ from the actual rating files.")
        report["review_context"] = context
        report["review_context_sha256"] = file_hash(context_path)
    if (output / "selection.json").exists():
        raise FileExistsError("A selection record already exists; retain it and version any protocol revision.")
    write_json(output / "selection.json", report)
    return report
