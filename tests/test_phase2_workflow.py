import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest

from PIL import Image
import torch

from aurora.app import create_app, default_bundle
from aurora.evaluation.rgb import evaluate_bundle, evaluate_candidates, finalize_selection, rank_candidates
from aurora.inference import RGBInference
from aurora.manifest import file_hash
from aurora.models import ResidualUNet
from aurora.training.run import save_bundle
from aurora.worker import WorkerManager

ROOT = Path(__file__).resolve().parents[1]


class Phase2Workflow(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp = TemporaryDirectory(dir=ROOT / "artifacts/tests")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rows = []
        for index, split in enumerate(["train", "diagnostic", "diagnostic", "selection", "selection", "test"]):
            raw, ref = self.root / f"raw-{index}.png", self.root / f"ref-{index}.png"
            Image.new("RGB", (19+index, 17+index), (40+index, 80+index, 120+index)).save(raw)
            Image.new("RGB", (19+index, 17+index), (70+index, 100+index, 140+index)).save(ref)
            rows.append({"id": str(index), "scene_id": str(index), "split": split,
                         "raw_path": raw.name, "reference_path": ref.name,
                         "raw_sha256": file_hash(raw), "reference_sha256": file_hash(ref)})
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps({"status": "frozen", "pairs": rows, "challenging": []}))
        manifest.with_suffix(".json.sha256").write_text(file_hash(manifest))
        self.config = json.loads((ROOT / "configs/rgb_baseline.json").read_text()) | {
            "manifest": str(manifest), "manifest_sha256": file_hash(manifest), "data_root": str(self.root),
            "epochs": 2, "candidate_epochs": [1, 2], "batch_size": 1,
            "model": {"base_channels": 4, "residual_scale": .5}}
        self.run = self.root / "run"
        self.run.mkdir()
        (self.run / "config.json").write_text(json.dumps(self.config))
        torch.manual_seed(123)
        model = ResidualUNet(**self.config["model"])
        optimizer = torch.optim.AdamW(model.parameters())
        for epoch in [1, 2]:
            with torch.no_grad():
                model.head.bias.fill_(.02*epoch)
            save_bundle(model, optimizer, self.run / f"epoch-{epoch:03d}", config=self.config,
                        epoch=epoch, step=epoch, provenance={"fixture_only": True})
        (self.run / "status.json").write_text(json.dumps({"status": "complete", "run_kind": "baseline_candidate",
            "steps": 2, "candidate_bundles": ["epoch-001", "epoch-002"]}))
        self.bundle = self.run / "epoch-002/bundle.json"

    def test_matched_identity_pilot_and_diagnostic_only_guard(self):
        from aurora.training.pilot import run_identity_pilot
        common = self.config | {"run_kind": "diagnostic_pilot", "protocol_version": "fixture-pilot-v1",
            "threads": 2, "crop_size": 16, "l1_prediction": "unclamped", "augmentation_epoch_offset": 2,
            "init_bundle": str(self.bundle), "init_weights_sha256": file_hash(self.bundle.parent / "weights.pt")}
        paths = {}
        for name, weight in [("control", 0.0), ("identity", .5)]:
            path = self.root / f"{name}.json"
            path.write_text(json.dumps(common | {"output": str(self.root / name), "identity_weight": weight}))
            paths[name] = str(path)
        protocol = {"schema_version": 1, "type": "diagnostic_identity_pilot", "arms": paths,
            "identity_weight": .5, "source_weights_sha256": common["init_weights_sha256"],
            "identity_diagnostic_ids": ["1", "2"], "output": str(self.root / "pilot")}
        path = self.root / "pilot.json"
        path.write_text(json.dumps(protocol))
        result = run_identity_pilot(path)
        self.assertTrue(result["training_order_matched"])
        self.assertFalse(result["selected"])
        self.assertEqual(result["selection_images_used"], 0)
        for name in paths:
            self.assertEqual(result["results"][name]["training"]["steps"], 2)
            self.assertTrue(result["results"][name]["training"]["checkpoint_reload_exact"])
            self.assertEqual(result["results"][name]["diagnostic_count"], 2)
        provenance = json.loads((self.root / "identity/provenance.json").read_text())
        self.assertEqual(provenance["initialization"]["weights_sha256"], common["init_weights_sha256"])
        self.assertEqual(provenance["initialization"]["source_epoch"], 2)
        record = json.loads((self.root / "identity/steps.jsonl").read_text().splitlines()[0])
        self.assertGreater(record["identity_loss"], 0)
        self.assertAlmostEqual(record["loss"], record["reconstruction_loss"]+.5*record["identity_loss"], places=7)
        with self.assertRaisesRegex(ValueError, "cannot enter selection"):
            evaluate_bundle(self.root / "identity/last/bundle.json", "selection", self.root / "forbidden")
        self.assertEqual(run_identity_pilot(path), result)
        # Centered follow-up reuses the original control and checksummed evaluation.
        from aurora.training.pilot import reused_diagnostic
        cached_path = self.root / "pilot/evaluation/control/summary.json"
        reuse = {"summary": str(cached_path), "summary_sha256": file_hash(cached_path)}
        source_bundle = json.loads((self.root / "control/last/bundle.json").read_text())
        self.assertEqual(reused_diagnostic(reuse, source_bundle)["count"], 2)
        with self.assertRaisesRegex(ValueError, "checksum"):
            reused_diagnostic(reuse | {"summary_sha256": "0"*64}, source_bundle)
        original_files = {str(p):file_hash(p) for folder in [self.root / "control", cached_path.parent] for p in folder.rglob("*") if p.is_file()}
        centered_path = self.root / "centered.json"
        centered_path.write_text(json.dumps(json.loads(Path(paths["identity"]).read_text()) |
                                  {"identity_mode": "centered_residual", "output": str(self.root / "centered")}))
        followup = protocol | {"treatment_variable": "centered_reference_identity",
            "arms": {"control": paths["control"], "identity": str(centered_path)},
            "reuse_diagnostic_evaluation": {"control": reuse}, "output": str(self.root / "followup")}
        followup_path = self.root / "followup.json"
        followup_path.write_text(json.dumps(followup))
        followup_result = run_identity_pilot(followup_path)
        self.assertTrue(followup_result["results"]["control"]["diagnostic_evaluation_reused"])
        self.assertTrue(followup_result["training_order_matched"])
        self.assertEqual(original_files, {p:file_hash(p) for p in original_files})
        changed = json.loads(Path(paths["identity"]).read_text()) | {"learning_rate": .123}
        Path(paths["identity"]).write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "differ in more"):
            run_identity_pilot(path)

    def test_valid_center_pilot_reuses_control_and_preserves_native_evaluation(self):
        from aurora.training.run import train
        from aurora.training.pilot import run_context_pilot, run_identity_pilot
        base = self.config | {"run_kind":"diagnostic_pilot", "protocol_version":"fixture-context",
            "threads":2, "crop_size":16, "l1_prediction":"unclamped", "augmentation_epoch_offset":2,
            "identity_weight":0., "init_bundle":str(self.bundle),
            "init_weights_sha256":file_hash(self.bundle.parent / "weights.pt")}
        configs = {}
        for name, margin in [("control",0),("treatment",2)]:
            cfg = base | {"output":str(self.root / name)}
            if margin: cfg["loss_margin"] = margin
            path = self.root / (name+".json"); path.write_text(json.dumps(cfg)); configs[name]=str(path)
        train(configs["control"])
        cached = self.root / "cached"
        evaluate_bundle(self.root / "control/last/bundle.json", "diagnostic", cached, threads=2)
        frozen = {str(p):file_hash(p) for folder in [cached,self.root/"control"] for p in folder.rglob("*") if p.is_file()}
        protocol = {"schema_version":1,"type":"diagnostic_valid_center_pilot", "arms":configs,
            "loss_margin":2, "identity_weight":0., "source_weights_sha256":base["init_weights_sha256"],
            "native_edge_ids":["1","2"], "native_edge_band":2,
            "reuse_diagnostic_evaluation":{"control":{"summary":str(cached/"summary.json"),"summary_sha256":file_hash(cached/"summary.json")}},
            "output":str(self.root / "context")}
        path = self.root / "protocol.json"; path.write_text(json.dumps(protocol))
        result = run_context_pilot(path)
        self.assertTrue(result["training_order_matched"])
        self.assertTrue(result["results"]["control"]["diagnostic_evaluation_reused"])
        self.assertEqual(result["results"]["treatment"]["supervised_size"],12)
        self.assertEqual(result["results"]["treatment"]["training"]["steps"],2)
        self.assertTrue(result["results"]["treatment"]["training"]["checkpoint_reload_exact"])
        self.assertEqual(result["results"]["treatment"]["native_edges"]["count"],2)
        self.assertNotIn("identity_diagnostics",result["results"]["treatment"])
        summary = json.loads((self.root / "context/evaluation/treatment/summary.json").read_text())
        self.assertEqual(summary["records"][0]["width"],20)
        self.assertEqual(summary["records"][0]["height"],18)
        self.assertEqual(frozen,{p:file_hash(p) for p in frozen})
        with self.assertRaisesRegex(ValueError,"cannot enter selection"):
            evaluate_bundle(self.root/"treatment/last/bundle.json","selection",self.root/"forbidden")
        with self.assertRaisesRegex(ValueError,"identity diagnostic"):
            run_identity_pilot(path)
        changed = json.loads(Path(configs["treatment"]).read_text()) | {"loss_margin":1}
        Path(configs["treatment"]).write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError,"declared reconstruction margin"):
            run_context_pilot(path)

    def test_pilot_initialization_and_objective_guards(self):
        from aurora.training.run import train, validate_config
        with self.assertRaisesRegex(ValueError, "restricted"):
            validate_config(self.config | {"identity_weight": .5})
        cfg = self.config | {"run_kind": "diagnostic_pilot", "protocol_version": "fixture",
            "threads": 2, "crop_size": 16, "output": str(self.root / "bad-pilot"),
            "init_bundle": str(self.bundle), "init_weights_sha256": "0"*64}
        path = self.root / "bad.json"
        path.write_text(json.dumps(cfg))
        with self.assertRaisesRegex(ValueError, "initialization differs"):
            train(path)
        with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
            validate_config(cfg | {"identity_weight": float("nan")})

    def test_native_evaluation_resume_and_source_tampering(self):
        output = self.root / "evaluation"
        first = evaluate_bundle(self.bundle, "diagnostic", output, threads=2)
        second = evaluate_bundle(self.bundle, "diagnostic", output, threads=2)
        self.assertEqual(first, second)
        self.assertEqual(first["count"], 2)
        self.assertEqual((first["records"][0]["width"], first["records"][0]["height"]), (20, 18))
        cached = output / "records/1.json"
        original = cached.read_text()
        altered = json.loads(original)
        altered["record"]["psnr"] = 99
        cached.write_text(json.dumps(altered))
        with self.assertRaisesRegex(ValueError, "checksum"):
            evaluate_bundle(self.bundle, "diagnostic", output, threads=2)
        cached.write_text(original)
        (self.root / "raw-1.png").write_bytes(b"corrupt")
        with self.assertRaises(ValueError):
            evaluate_bundle(self.bundle, "diagnostic", output, threads=2)

    def test_final_holdouts_and_smoke_selection_rejected(self):
        for split in ["test", "challenging", "train"]:
            with self.assertRaises(ValueError):
                evaluate_bundle(self.bundle, split, self.root / split)
        from aurora.image import validate_image
        with self.assertRaisesRegex(ValueError, "sealed"):
            RGBInference(self.bundle).enhance(validate_image(self.root / "raw-5.png"))
        metadata = json.loads(self.bundle.read_text())
        metadata["run_kind"] = "training_smoke_only"
        self.bundle.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "smoke"):
            evaluate_bundle(self.bundle, "selection", self.root / "smoke")

    def test_checkpoint_tie_breaking(self):
        a = {"mean_psnr": 20., "mean_ssim": .8, "training_steps": 10}
        b = {"mean_psnr": 20.01, "mean_ssim": .7, "training_steps": 20}
        self.assertIs(rank_candidates([a, b]), a)
        c = {"mean_psnr": 20.011, "mean_ssim": .6, "training_steps": 30}
        self.assertIs(rank_candidates([a, c]), c)
        self.assertIs(rank_candidates([a, a | {"training_steps": 20}]), a)

    def test_diagnostic_only_evaluation_does_not_open_selection(self):
        from unittest.mock import patch
        output = self.root / "diagnostic_only"
        with patch("aurora.evaluation.rgb.evaluate_bundle", wraps=evaluate_bundle) as evaluator:
            result = evaluate_candidates(self.run, output, threads=2, diagnostic_only=True)
        self.assertEqual(result["status"], "diagnostic_only_complete")
        self.assertEqual(result["selection_images_used"], 0)
        self.assertTrue(all(call.args[1] == "diagnostic" for call in evaluator.call_args_list))
        self.assertFalse((output / "ledger.json").exists())
        self.assertFalse((output / "review").exists())
        for name in ["epoch-001", "epoch-002"]:
            self.assertFalse((output / name / "selection").exists())

    def fill_reviews(self, output, ledger, severity="none", same_person=False):
        for candidate in ledger["candidates"]:
            for index in [1, 2]:
                path = output / "review" / candidate["label"] / f"reviewer-{index}.csv"
                with path.open(newline="") as stream:
                    rows = list(csv.DictReader(stream))
                for row in rows:
                    row.update(reviewer=f"Fixture reviewer {1 if same_person else index}", independent="yes", severity=severity,
                               notes="Synthetic test rating" if severity != "none" else "")
                with path.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)

    def test_review_gate_and_complete_selection(self):
        output = self.root / "artifacts/phase2/evaluation"
        ledger = evaluate_candidates(self.run, output, threads=2)
        self.assertFalse(ledger["selected"])
        with self.assertRaises(ValueError):
            finalize_selection(output)
        self.fill_reviews(output, ledger, same_person=True)
        with self.assertRaisesRegex(ValueError, "distinct"):
            finalize_selection(output)
        self.fill_reviews(output, ledger)
        context_path = output / "review_context.json"
        context_path.write_text(json.dumps({"reviewer_ids": ["wrong", "identities"], "review_type": "synthetic fixture"}))
        with self.assertRaisesRegex(ValueError, "context identities"):
            finalize_selection(output)
        context = {"reviewer_ids": ["Fixture reviewer 1", "Fixture reviewer 2"], "review_type": "synthetic fixture"}
        context_path.write_text(json.dumps(context))
        selected = finalize_selection(output)
        self.assertTrue(selected["selected"])
        self.assertEqual(selected["review_context"], context)
        self.assertEqual(selected["review_context_sha256"], file_hash(context_path))
        self.assertEqual(selected["chosen"]["name"], ledger["provisional_metric_leader"])
        self.assertEqual(default_bundle(self.root), (selected["chosen"]["bundle"], True))
        with self.assertRaises(FileExistsError):
            finalize_selection(output)

    def test_major_artifacts_prevent_selection(self):
        output = self.root / "artifacts/phase2/evaluation"
        ledger = evaluate_candidates(self.run, output, threads=2)
        self.fill_reviews(output, ledger, severity="major")
        self.assertFalse(finalize_selection(output)["selected"])
        self.assertEqual(default_bundle(self.root), (None, False))

    def test_experimental_default_requires_pinned_decision_bundle_and_weights(self):
        cfg = self.root / 'configs'; cfg.mkdir()
        bundle = json.loads(self.bundle.read_text())
        decision = self.root / 'decision.json'
        decision.write_text(json.dumps({'status':'user_authorized_experimental_prototype',
            'bundle':str(self.bundle.relative_to(self.root)), 'bundle_sha256':file_hash(self.bundle),
            'weights_sha256':bundle['weights_sha256']}))
        active = cfg / 'experimental_model.json'
        active.write_text(json.dumps({'record':'decision.json','record_sha256':file_hash(decision)}))
        self.assertEqual(default_bundle(self.root), (str(self.bundle), False))
        weights = self.bundle.parent / bundle['weights']
        original = weights.read_bytes(); weights.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Experimental model identity'):
            default_bundle(self.root)
        weights.write_bytes(original)
        decision.write_text(decision.read_text()+' ')
        with self.assertRaisesRegex(ValueError, 'decision identity'):
            default_bundle(self.root)

    def manager(self, bundle=None):
        manager = WorkerManager(bundle or self.bundle, root=self.root / "jobs")
        self.addCleanup(manager.close)
        return manager

    def wait_for_result(self, manager, token):
        until = time.monotonic()+20
        while time.monotonic() < until:
            record = manager.poll("session", token)
            if record["state"] in {"success", "failed"}:
                return record
            time.sleep(.05)
        self.fail("Worker did not finish within the quick-check timeout")

    def test_worker_cancel_retry_new_upload_and_png(self):
        manager = self.manager()
        first = manager.upload("session", self.root / "raw-1.png")
        manager.start("session", first.token)
        before = time.monotonic()
        cancelled = manager.cancel("session", first.token)
        self.assertLess(time.monotonic()-before, 2)
        self.assertEqual(cancelled["state"], "cancelled")
        self.assertIsNone(cancelled["output"])
        self.assertIsNone(manager.sessions["session"].process)
        manager.start("session", first.token)
        result = self.wait_for_result(manager, first.token)
        self.assertEqual(result["state"], "success", result)
        with Image.open(result["output"]) as exported:
            self.assertEqual(exported.size, (20, 18))
            self.assertFalse(exported.getexif())
            self.assertIn("icc_profile", exported.info)
        old = Path(result["output"])
        replacement = manager.upload("session", self.root / "raw-2.png")
        self.assertFalse(old.exists())
        self.assertEqual(manager.poll("session", first.token)["state"], "stale")
        self.assertIsNone(manager.poll("session", replacement.token)["output"])
        manager.start("session", replacement.token)
        self.assertEqual(self.wait_for_result(manager, replacement.token)["state"], "success")
        broken = self.root / "broken.png"
        broken.write_bytes(b"broken")
        invalid = manager.upload("session", broken)
        self.assertIsNone(invalid.valid)
        self.assertIsNone(manager.poll("session", invalid.token)["output"])
        self.assertFalse(manager.poll("session", invalid.token)["valid_input"])

    def test_missing_weights_clear_failure_and_app_contract(self):
        manager = self.manager()
        (self.bundle.parent / "weights.pt").unlink()
        session = manager.upload("session", self.root / "raw-1.png")
        manager.start("session", session.token)
        result = self.wait_for_result(manager, session.token)
        self.assertEqual(result["state"], "failed")
        self.assertIsNone(result["output"])
        app = create_app(str(self.bundle))
        self.addCleanup(app.aurora_manager.close)
        app.queue().validate_queue_settings()
        components = app.get_config_file()["components"]
        self.assertEqual(len([c for c in components if c["type"] == "image"]), 2)
        self.assertEqual(len([c for c in components if c["type"] == "downloadbutton"]), 1)

    def test_padding_cost_and_idle_cleanup(self):
        manager = self.manager()
        oversized = self.root / "thin.png"
        Image.new("RGB", (400000, 1)).save(oversized)
        rejected = manager.upload("session", oversized)
        self.assertIsNone(rejected.valid)
        self.assertIn("padding", rejected.message)
        session = manager.upload("session", self.root / "raw-1.png")
        manager.start("session", session.token)
        directory = session.directory
        session.touched = time.monotonic()-100
        manager.reap_expired()
        self.assertNotIn("session", manager.sessions)
        self.assertIsNone(session.process)
        self.assertFalse(directory.exists())
        self.assertEqual(manager.poll("session", session.token)["state"], "expired")


if __name__ == "__main__":
    unittest.main()
