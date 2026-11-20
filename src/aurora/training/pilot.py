"""Matched diagnostic-only fine-tuning pilot; no selection or release approval."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from ..evaluation.rgb import evaluate_bundle
from ..image import validate_image
from ..inference import RGBInference
from ..manifest import file_hash, read_manifest, safe_path
from .metrics import METRIC_VERSION
from .run import train, validate_config, write_json


def sequence_hash(path):
    digest = hashlib.sha256()
    with Path(path).open() as stream:
        for line in stream:
            record = json.loads(line)
            digest.update(json.dumps([record["step"], record["epoch"], record["ids"]], separators=(",", ":")).encode())
    return digest.hexdigest()


def identity_diagnostics(bundle, ids):
    model = RGBInference(bundle)
    config = model.bundle["configuration"]
    manifest = read_manifest(config["manifest"])
    rows = {r["id"]: r for r in manifest["pairs"] if r["split"] == "diagnostic"}
    results = []
    for identifier in ids:
        if identifier not in rows:
            raise ValueError("Identity diagnostic is outside the diagnostic split.")
        row = rows[identifier]
        valid = validate_image(safe_path(config["data_root"], row["reference_path"]))
        if valid.source_sha256 != row["reference_sha256"]:
            raise ValueError("Identity diagnostic reference changed after freeze.")
        reference = torch.from_numpy(np.array(valid.image, dtype=np.float32).transpose(2, 0, 1).copy()/255).unsqueeze(0)
        prediction = model.predict(valid)
        delta = prediction-reference
        centered = delta-delta.mean(dim=(-2, -1), keepdim=True)
        results.append({"id": identifier, "mean_absolute_identity_change": (prediction-reference).abs().mean().item(),
                        "mean_centered_absolute_identity_change": centered.abs().mean().item(),
                        "fraction_changed_gt_one_level": ((prediction-reference).abs()>1/255).float().mean().item()})
    return results


def reused_diagnostic(spec, bundle):
    """Read and validate the prior control evaluation without modifying it."""
    path = Path(spec["summary"])
    if file_hash(path) != spec["summary_sha256"]:
        raise ValueError("Reused diagnostic summary checksum mismatch.")
    summary = json.loads(path.read_text())
    cfg = bundle["configuration"]
    manifest = read_manifest(cfg["manifest"])
    rows = {r["id"]: r for r in manifest["pairs"] if r["split"] == "diagnostic"}
    if (not summary["complete"] or summary["split"] != "diagnostic" or
            summary["weights_sha256"] != bundle["weights_sha256"] or
            summary["manifest_sha256"] != cfg["manifest_sha256"] or
            file_hash(cfg["manifest"]) != cfg["manifest_sha256"] or
            summary["metric_version"] != METRIC_VERSION or summary["count"] != len(rows) or
            len(summary["records"]) != len(rows) or {r["id"] for r in summary["records"]} != set(rows)):
        raise ValueError("Reused evaluation differs in weights, data, split or metrics.")
    for record in summary["records"]:
        row = rows[record["id"]]
        for key in ["raw", "reference"]:
            if record[key+"_sha256"] != row[key+"_sha256"] or file_hash(safe_path(cfg["data_root"], row[key+"_path"])) != row[key+"_sha256"]:
                raise ValueError("Reused evaluation source mismatch.")
        if file_hash(safe_path(path.parent / "images", record["id"]+".png")) != record["png_sha256"]:
            raise ValueError("Reused evaluation PNG checksum mismatch.")
    return summary


def native_edge_diagnostics(summary, evaluation_path, config, ids, band):
    """Supplementary PNG MSE on fixed native edges; no additional inference."""
    if type(band) is not int or band < 1 or not ids or len(set(ids)) != len(ids):
        raise ValueError("Native-edge diagnostic requires unique fixed cases and a positive band.")
    rows = {r["id"]:r for r in read_manifest(config["manifest"])["pairs"] if r["split"] == "diagnostic"}
    records = {r["id"]:r for r in summary["records"]}
    results = []
    for identifier in ids:
        if identifier not in rows or identifier not in records:
            raise ValueError("Native-edge case is outside diagnostic evaluation.")
        row = rows[identifier]
        reference = validate_image(safe_path(config["data_root"], row["reference_path"]))
        path = safe_path(Path(evaluation_path) / "images", identifier+".png")
        if reference.source_sha256 != row["reference_sha256"] or file_hash(path) != records[identifier]["png_sha256"]:
            raise ValueError("Native-edge source/output checksum mismatch.")
        with Image.open(path) as image:
            predicted = np.asarray(image.convert("RGB"), dtype=np.float64)/255
        target = np.asarray(reference.image, dtype=np.float64)/255
        if predicted.shape != target.shape or min(target.shape[:2]) <= 2*band:
            raise ValueError("Invalid native-edge geometry/band.")
        squared = (predicted-target)**2
        interior = squared[band:-band, band:-band]
        results.append({"id": identifier, "border_mse": float((squared.sum()-interior.sum())/(squared.size-interior.size)),
                        "interior_mse": float(interior.mean())})
    return {"metric": "decoded-png-rgb-mse-v1", "band_pixels": band, "count": len(results), "records": results,
            "mean_border_mse": float(np.mean([r["border_mse"] for r in results])),
            "mean_interior_mse": float(np.mean([r["interior_mse"] for r in results]))}


def run_identity_pilot(protocol_path):
    if json.loads(Path(protocol_path).read_text()).get("type") != "diagnostic_identity_pilot":
        raise ValueError("identity-pilot requires an identity diagnostic protocol.")
    return run_diagnostic_pilot(protocol_path)


def run_context_pilot(protocol_path):
    if json.loads(Path(protocol_path).read_text()).get("type") != "diagnostic_valid_center_pilot":
        raise ValueError("context-pilot requires a valid-center diagnostic protocol.")
    return run_diagnostic_pilot(protocol_path)


def run_diagnostic_pilot(protocol_path):
    protocol_path = Path(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("schema_version") != 1 or protocol.get("type") not in {"diagnostic_identity_pilot", "diagnostic_valid_center_pilot"}:
        raise ValueError("Unsupported diagnostic pilot protocol.")
    context = protocol["type"] == "diagnostic_valid_center_pilot"
    treatment = "treatment" if context else "identity"
    if set(protocol["arms"]) != {"control", treatment}:
        raise ValueError("The fixed pilot requires one control and one declared treatment arm.")
    configs = {name: json.loads(Path(path).read_text()) for name, path in protocol["arms"].items()}
    for config in configs.values():
        validate_config(config)
        if config["run_kind"] != "diagnostic_pilot":
            raise ValueError("Pilot arms must not masquerade as baseline candidates.")
    a, b = configs["control"], configs[treatment]
    ignored = {"identity_weight", "output"}
    if context:
        ignored.add("loss_margin")
        if (a.get("loss_margin", 0) != 0 or b.get("loss_margin") != protocol["loss_margin"] or
                type(protocol["loss_margin"]) is not int or protocol["loss_margin"] <= 0 or
                a.get("identity_weight", 0) != 0 or b.get("identity_weight", 0) != 0 or
                a.get("identity_mode", "full") != "full" or b.get("identity_mode", "full") != "full"):
            raise ValueError("Valid-center pilot must change only the declared reconstruction margin with identity disabled.")
    elif protocol.get("treatment_variable") == "centered_reference_identity":
        ignored.add("identity_mode")
        if a.get("identity_mode", "full") != "full" or b.get("identity_mode") != "centered_residual":
            raise ValueError("Centered pilot must declare the reference residual centering treatment.")
    elif a.get("identity_mode", "full") != b.get("identity_mode", "full"):
        raise ValueError("Identity mode change requires a declared centered-reference protocol.")
    if {k:v for k,v in a.items() if k not in ignored} != {k:v for k,v in b.items() if k not in ignored}:
        raise ValueError("Pilot arms differ in more than the declared treatment/output location.")
    if a.get("identity_weight") != 0 or b.get("identity_weight") != protocol["identity_weight"] or a["output"] == b["output"]:
        raise ValueError("Pilot control/treatment declaration does not match the protocol.")
    initial = RGBInference(a["init_bundle"])
    if (initial.bundle["weights_sha256"] != a["init_weights_sha256"] or
            initial.bundle["weights_sha256"] != protocol["source_weights_sha256"] or
            initial.bundle["model"] != a["model"] or
            initial.bundle["configuration"]["manifest_sha256"] != a["manifest_sha256"]):
        raise ValueError("Pilot source weights differ from the predeclared initialization.")
    source_identity = initial.bundle["weights_sha256"]
    del initial
    output = Path(protocol["output"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "protocol.json").exists() and json.loads((output / "protocol.json").read_text()) != protocol:
        raise ValueError("Existing pilot output belongs to another protocol.")
    write_json(output / "protocol.json", protocol)
    results = {}
    for name in ["control", treatment]:
        config = configs[name]
        run = Path(config["output"])
        if run.exists():
            if not (run / "status.json").exists() or json.loads((run / "config.json").read_text()) != config:
                raise ValueError("Existing pilot run is incomplete or has different configuration; preserve it and use a new declared output path.")
            training = json.loads((run / "status.json").read_text())
            if training["status"] != "complete":
                raise ValueError("Mid-run resume is unsupported. Preserve the interrupted pilot and declare new output paths.")
        else:
            print(f"Training diagnostic pilot arm: {name}", flush=True)
            training = train(protocol["arms"][name])
        finished = RGBInference(run / "last/bundle.json")
        if finished.bundle["configuration"] != config or finished.bundle["weights_sha256"] != training["weights_sha256"]:
            raise ValueError("Completed pilot checkpoint/configuration/status identities differ.")
        finished_bundle = finished.bundle
        del finished
        print(f"Evaluating diagnostic images only: {name}", flush=True)
        torch.set_num_threads(config["threads"])
        reuse = protocol.get("reuse_diagnostic_evaluation", {}).get(name)
        if reuse:
            evaluated = reused_diagnostic(reuse, finished_bundle)
        else:
            evaluated = evaluate_bundle(run / "last/bundle.json", "diagnostic", output / "evaluation" / name,
                                        threads=config["threads"])
        results[name] = {"training": training, "training_sequence_sha256": sequence_hash(run / "steps.jsonl"),
                         "diagnostic_count": evaluated["count"], "diagnostic_mean_psnr": evaluated["mean_psnr"],
                         "diagnostic_mean_ssim": evaluated["mean_ssim"], "diagnostic_evaluation_reused": bool(reuse)}
        if context:
            results[name].update(loss_margin=config.get("loss_margin", 0),
                                 supervised_size=config["crop_size"]-2*config.get("loss_margin", 0))
            evaluation_path = Path(reuse["summary"]).parent if reuse else output / "evaluation" / name
            edges = native_edge_diagnostics(evaluated, evaluation_path, config,
                                            protocol["native_edge_ids"], protocol["native_edge_band"])
            write_json(output / f"native_edges_{name}.json", edges)
            results[name]["native_edges"] = {k:v for k,v in edges.items() if k != "records"}
        else:
            identity = identity_diagnostics(run / "last/bundle.json", protocol["identity_diagnostic_ids"])
            results[name].update(identity_diagnostics=identity,
                mean_centered_identity_change=float(np.mean([r["mean_centered_absolute_identity_change"] for r in identity])),
                mean_identity_change=float(np.mean([r["mean_absolute_identity_change"] for r in identity])))
        write_json(output / "progress.json", {"status": "in_progress", "completed_arms": list(results), "results": results})
    if results["control"]["training_sequence_sha256"] != results[treatment]["training_sequence_sha256"]:
        raise ValueError("Pilot arms did not use identical training ID/epoch sequences.")
    report = {"status": "diagnostic_pilot_complete", "selected": False,
              "source_weights_sha256": source_identity, "protocol_sha256": file_hash(protocol_path),
              "results": results, "training_order_matched": True, "selection_images_used": 0,
              "test_images_used": 0, "challenging_images_used": 0, "phase2_complete": False,
              "notice": ("Valid-center diagnostic comparison with full native evaluation, not a release baseline. Inspect native image edges and central fidelity; reduced supervision is a confound." if context else
                         "Two diagnostic fine-tuning arms, not a trained release baseline or formal candidate selection. Evaluate enhancement/identity tradeoffs and artifacts before declaring further training.")}
    write_json(output / "summary.json", report)
    return report
