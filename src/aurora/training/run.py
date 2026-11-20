"""CPU reconstruction runs with immutable configuration and per-step evidence."""
import json
import math
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..dataset import PairedDataset
from ..image import PREPROCESS_VERSION
from ..manifest import file_hash, read_manifest
from ..models import ARCHITECTURES, SCENE_ARCHITECTURE, DEPTH_ARCHITECTURE, DEPTH_ARCHITECTURES, build_rgb
from .metrics import METRIC_VERSION, ssim


def write_json(path, document):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def validate_config(config):
    if config["schema_version"] != 1 or config["architecture"] not in ARCHITECTURES:
        raise ValueError("Unsupported training configuration or architecture.")
    if config["device"] != "cpu":
        raise ValueError("Only the validated CPU training profile is currently implemented.")
    if config["run_kind"] not in {"training_smoke_only", "baseline_candidate", "diagnostic_pilot"}:
        raise ValueError("Unknown run kind.")
    if config.get("l1_prediction", "bounded") not in {"bounded", "unclamped"}:
        raise ValueError("l1_prediction must be bounded or unclamped.")
    if config['architecture'] in DEPTH_ARCHITECTURES and not all(config.get(k) for k in ['depth_config', 'depth_config_sha256', 'depth_fingerprint']):
        raise ValueError('Depth training requires a pinned depth configuration and fingerprint.')
    identity_weight = config.get("identity_weight", 0.0)
    scene_model = config['architecture'] in ({SCENE_ARCHITECTURE} | DEPTH_ARCHITECTURES)
    if scene_model:
        if config.get('scene_context') != 'native-input-channel-moments-v1':
            raise ValueError('Scene architecture requires native input scene_context.')
        if identity_weight or config.get('loss_margin', 0) or config.get('init_bundle'):
            raise ValueError('Scene baseline requires fresh initialization and full-crop reconstruction without identity loss.')
        if config.get('l1_prediction') != 'unclamped':
            raise ValueError('Scene baseline requires unclamped L1.')
    elif config.get('scene_context'):
        raise ValueError('Scene context requires the scene architecture.')
    margin = config.get("loss_margin", 0)
    if type(margin) is not int or margin < 0:
        raise ValueError("loss_margin must be a nonnegative integer.")
    if margin and identity_weight:
        raise ValueError("Valid-center reconstruction and identity loss must be separate diagnostic experiments.")
    if config.get("identity_mode", "full") not in {"full", "centered_residual"}:
        raise ValueError("Unknown reference identity mode.")
    if not math.isfinite(identity_weight) or identity_weight < 0:
        raise ValueError("identity_weight must be finite and nonnegative.")
    if type(config.get("augmentation_epoch_offset", 0)) is not int or config.get("augmentation_epoch_offset", 0) < 0:
        raise ValueError("augmentation_epoch_offset must be a nonnegative integer.")
    if (margin or identity_weight or config.get("init_bundle") or config.get("identity_mode", "full") != "full") and config["run_kind"] != "diagnostic_pilot":
        raise ValueError("Warm-start, loss margins and identity loss are restricted to a declared diagnostic_pilot.")
    if config["run_kind"] == "diagnostic_pilot" and not all(config.get(k) for k in ["init_bundle", "init_weights_sha256", "protocol_version"]):
        raise ValueError("Diagnostic pilot requires a versioned protocol and pinned initialization bundle.")
    for key in ["seed", "batch_size", "epochs", "crop_size", "threads"]:
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    if config["crop_size"] < 11:
        raise ValueError("SSIM requires crops of at least 11 pixels.")
    if config["crop_size"] - 2*margin < 11:
        raise ValueError("Valid-center supervision must leave at least 11 pixels per axis for SSIM.")
    weights = config["loss_weights"]
    if set(weights) != {"l1", "ssim", "perceptual", "adversarial"} or not all(
        math.isfinite(v) for v in weights.values()
    ) or weights["l1"] <= 0 or weights["ssim"] <= 0 or weights["perceptual"] != 0 or weights["adversarial"] != 0:
        raise ValueError("Reconstruction requires active L1/SSIM and zero perceptual/adversarial losses.")
    for key in ["learning_rate", "weight_decay"]:
        if not math.isfinite(config[key]) or config[key] < 0 or (key == "learning_rate" and config[key] == 0):
            raise ValueError(f"Invalid {key}.")
    candidates = config["candidate_epochs"]
    if candidates != sorted(set(candidates)) or any(type(e) is not int or not 1 <= e <= config["epochs"] for e in candidates):
        raise ValueError("Candidate epochs must be unique, increasing and within the run.")
    if "max_steps" in config and (config["run_kind"] != "training_smoke_only" or
                                  type(config["max_steps"]) is not int or config["max_steps"] < 1):
        raise ValueError("Only a labeled training smoke run may have a positive max_steps cap.")
    if config["run_kind"] == "baseline_candidate":
        if not candidates or not all(config.get(k) for k in ["candidate_search_budget", "selection_policy", "review_rubric"]):
            raise ValueError("Baseline runs require a declared candidate budget, selection policy and rubric.")


def reference_identity_loss(raw_prediction, reference, mode="full"):
    residual = raw_prediction - reference
    if mode == "centered_residual":
        residual = residual - residual.mean(dim=(-2, -1), keepdim=True)
    elif mode != "full":
        raise ValueError("Unknown reference identity mode.")
    return residual.abs().mean()


def reconstruction_terms(l1_prediction, bounded_prediction, reference, margin=0):
    if type(margin) is not int or margin < 0 or min(reference.shape[-2:])-2*margin < 11:
        raise ValueError("Invalid reconstruction supervision margin.")
    if l1_prediction.shape != reference.shape or bounded_prediction.shape != reference.shape:
        raise ValueError("Reconstruction prediction/reference shapes differ.")
    region = slice(margin, -margin) if margin else slice(None)
    l1_source = l1_prediction[..., region, region]
    bounded = bounded_prediction[..., region, region]
    target = reference[..., region, region]
    return (l1_source-target).abs().mean(), 1-ssim(bounded, target).mean()


def save_bundle(model, optimizer, directory, *, config, epoch, step, provenance):
    directory.mkdir()
    torch.save(model.state_dict(), directory / "weights.pt")
    # Separate optimizer state is retained for auditing, never loaded by inference.
    torch.save({"optimizer": optimizer.state_dict(), "epoch": epoch, "step": step}, directory / "training_state.pt")
    document = {"schema_version": 1, "architecture": config['architecture'], "model": model.settings,
                "preprocessing_version": PREPROCESS_VERSION, "metric_version": METRIC_VERSION,
                "weights": "weights.pt", "weights_sha256": file_hash(directory / "weights.pt"),
                "training_state_sha256": file_hash(directory / "training_state.pt"),
                "training_steps": step, "epoch": epoch, "run_kind": config["run_kind"],
                "selected": False, "configuration": config, "provenance": provenance}
    write_json(directory / "bundle.json", document)
    return document


def train(config_path):
    config_path = Path(config_path)
    config = json.loads(config_path.read_text())
    validate_config(config)
    manifest = read_manifest(config["manifest"])
    digest = file_hash(config["manifest"])
    if digest != config["manifest_sha256"]:
        raise ValueError("Training manifest differs from the predeclared configuration.")
    depth = None
    if config['architecture'] in DEPTH_ARCHITECTURES:
        from ..depth import DepthEstimator
        if file_hash(config['depth_config']) != config['depth_config_sha256']:
            raise ValueError('Depth configuration identity changed.')
        depth = DepthEstimator(config['depth_config'])
        if depth.fingerprint != config['depth_fingerprint']:
            raise ValueError('Depth estimator fingerprint changed.')
    data = PairedDataset(config["data_root"], config["manifest"], "train",
                         crop_size=config["crop_size"], seed=config["seed"],
                         depth=depth, include_scene_stats=config['architecture'] in ({SCENE_ARCHITECTURE} | DEPTH_ARCHITECTURES))
    if not len(data):
        raise ValueError("Training partition is empty.")
    output = Path(config["output"])
    # Existing runs are evidence. Reruns require an explicit new output path.
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    model = build_rgb(config['architecture'], config['model'])
    initialization = None
    if config.get("init_bundle"):
        from ..inference import RGBInference
        initial = RGBInference(config["init_bundle"])
        source = initial.bundle
        if source["weights_sha256"] != config["init_weights_sha256"] or source["model"] != config["model"] or source["configuration"]["manifest_sha256"] != digest:
            raise ValueError("Pilot initialization differs from pinned weights/model/data.")
        model.load_state_dict(initial.model.state_dict(), strict=True)
        initialization = {"bundle": config["init_bundle"], "bundle_sha256": file_hash(config["init_bundle"]),
                          "weights_sha256": source["weights_sha256"], "source_steps": source["training_steps"],
                          "source_epoch": source["epoch"], "source_run_kind": source["run_kind"],
                          "optimizer": "fresh AdamW; source optimizer state is deliberately not resumed"}
        del initial
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    generator = torch.Generator().manual_seed(config["seed"])
    loader = DataLoader(data, batch_size=config["batch_size"], shuffle=True,
                        generator=generator, num_workers=0)
    package = Path(__file__).resolve().parents[1]
    provenance = {"manifest_sha256": digest, "config_sha256": file_hash(config_path),
                  "lock_sha256": file_hash(package.parents[1] / "requirements.lock.txt"),
                  "source_sha256": {str(p.relative_to(package)): file_hash(p)
                                    for p in sorted(package.rglob("*.py"))},
                  "python": sys.version, "torch": str(torch.__version__), "numpy": np.__version__,
                  "hardware": platform.processor() or platform.machine(), "device": "cpu",
                  "parameter_count": sum(p.numel() for p in model.parameters()),
                  "training_images": len(data), "seed": config["seed"],
                  "augmentation": "synchronized crop, horizontal/vertical flips, 90-degree rotations; no photometric augmentation",
                  "optimizer": "AdamW; constant learning rate; no mixed precision on CPU",
                  "l1_prediction": config.get("l1_prediction", "bounded"),
                  "scene_context": config.get('scene_context'),
                  "identity_weight": config.get("identity_weight", 0.0),
                  "identity_mode": config.get("identity_mode", "full"),
                  "loss_margin": config.get("loss_margin", 0),
                  "supervised_size": config["crop_size"]-2*config.get("loss_margin", 0),
                  "initialization": initialization,
                  "augmentation_epoch_offset": config.get("augmentation_epoch_offset", 0),
                  "determinism": "Deterministic algorithms enabled; exact reproducibility is scoped to this CPU/software profile."}
    write_json(output / "provenance.json", provenance)
    start, step, losses = time.perf_counter(), 0, []
    checkpoints = []
    write_json(output / "status.json", {"status": "running", "steps": 0, "phase2_complete": False})
    try:
        with (output / "steps.jsonl").open("x") as log:
            for epoch in range(1, config["epochs"] + 1):
                data.set_epoch(epoch - 1 + config.get("augmentation_epoch_offset", 0))
                model.train()
                for batch in loader:
                    optimizer.zero_grad(set_to_none=True)
                    context = {'scene_stats': batch['scene_stats']} if 'scene_stats' in batch else {}
                    if depth is not None: context['depth'] = batch['depth']
                    if config.get("l1_prediction", "bounded") == "unclamped":
                        raw_prediction = model.forward_unclamped(batch["input"], **context)
                        prediction = raw_prediction.clamp(0, 1)
                        l1_source = raw_prediction
                    else:
                        prediction = model(batch["input"], **context)
                        l1_source = prediction
                    l1, structural = reconstruction_terms(l1_source, prediction, batch["reference"], config.get("loss_margin", 0))
                    reconstruction = config["loss_weights"]["l1"] * l1 + config["loss_weights"]["ssim"] * structural
                    identity = reconstruction.new_zeros(())
                    if config.get("identity_weight", 0) > 0:
                        identity_raw = model.forward_unclamped(batch["reference"])
                        identity = reference_identity_loss(identity_raw, batch["reference"], config.get("identity_mode", "full"))
                    loss = reconstruction + config.get("identity_weight", 0)*identity
                    if not torch.isfinite(loss):
                        raise ValueError("Non-finite reconstruction loss.")
                    loss.backward()
                    gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    step += 1
                    record = {"step": step, "epoch": epoch, "ids": batch["id"], "loss": loss.item(),
                              "l1": l1.item(), "ssim_loss": structural.item(), "gradient_norm": gradient.item(),
                              "reconstruction_loss": reconstruction.item(), "identity_loss": identity.item(),
                              "wall_seconds": time.perf_counter() - start}
                    log.write(json.dumps(record, allow_nan=False) + "\n")
                    log.flush()
                    losses.append(record["loss"])
                    print(json.dumps(record), flush=True)
                    if step % 25 == 0:
                        write_json(output / "status.json", {"status": "running", "steps": step,
                            "epoch": epoch, "total_epochs": config["epochs"],
                            "latest_loss": record["loss"], "wall_seconds": record["wall_seconds"],
                            "phase2_complete": False})
                    if config.get("max_steps") and step >= config["max_steps"]:
                        break
                if epoch in config["candidate_epochs"]:
                    name = f"epoch-{epoch:03d}"
                    save_bundle(model, optimizer, output / name, config=config, epoch=epoch, step=step, provenance=provenance)
                    checkpoints.append(name)
                if config.get("max_steps") and step >= config["max_steps"]:
                    break
        bundle = save_bundle(model, optimizer, output / "last", config=config, epoch=epoch, step=step, provenance=provenance)
        # Verify that the on-disk inference artifact reproduces the in-memory model.
        from ..inference import RGBInference
        restored = RGBInference(output / "last/bundle.json")
        model.eval()
        with torch.inference_mode():
            expected, actual = model(batch["input"], **context), restored.model(batch["input"], **context)
        if not torch.equal(expected, actual):
            raise ValueError("Checkpoint reload parity failed.")
        report = {"status": "complete", "run_kind": config["run_kind"], "steps": step,
                  "epochs_reached": epoch, "mean_training_loss": sum(losses) / len(losses),
                  "wall_seconds": time.perf_counter() - start, "finite_losses": True,
                  "checkpoint_reload_exact": True, "weights_sha256": bundle["weights_sha256"],
                  "candidate_bundles": checkpoints, "phase2_complete": False,
                  "quality_claim": "None; checkpoint selection and native-size artifact review are still required.",
                  "selection_images_used": 0, "test_images_used": 0, "challenging_images_used": 0}
        write_json(output / "status.json", report)
        return report
    except BaseException as exc:
        write_json(output / "status.json", {"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                   "steps": step, "error": str(exc), "phase2_complete": False})
        raise
