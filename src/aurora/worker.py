"""Interruptible local RGB jobs; operation tokens suppress late results."""
import atexit
from dataclasses import dataclass
import json
import logging
import multiprocessing as mp
from pathlib import Path
import re
import shutil
import tempfile
import threading
import time
import uuid

from .image import validate_image

APP_MAX_PIXELS = 2_500_000


def _write(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def _run(directory, bundle_path, token, source_hash, filename, max_pixels):
    """Spawn target: model is loaded once, with no UI-process torch state."""
    directory = Path(directory)
    status = directory / "status.json"
    started = time.perf_counter()
    try:
        _write(status, {"state": "loading", "token": token})
        import torch
        from .inference import RGBInference, export_png
        torch.set_num_threads(4)
        model = RGBInference(bundle_path, depth_policy='live')
        model_load_seconds = time.perf_counter()-started
        valid = validate_image(directory / "input.png", max_pixels=max_pixels)
        # This file is an already oriented, metadata-clean snapshot of the upload.
        valid.source_sha256 = source_hash
        _write(status, {"state": "processing", "token": token})
        image, report = model.enhance(valid, stage_callback=lambda stage: _write(status, {'state':stage,'token':token}))
        _write(status, {"state": "exporting", "token": token})
        export_png(image, directory / filename)
        import resource
        report.update(worker_seconds=time.perf_counter()-started,
                      model_load_seconds=model_load_seconds,
                      worker_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                      resource_scope="Linux CPU worker peak including model load, enhancement and PNG export; provisional observation, not a performance contract")
        _write(status, {"state": "success", "token": token, "filename": filename, "report": report,
                        "completed_monotonic": time.monotonic()})
    except BaseException as exc:
        logging.basicConfig(filename=str(directory.parent / "worker_errors.log"), level=logging.ERROR)
        logging.exception("RGB operation %s failed", token)
        if isinstance(exc, MemoryError) or "out of memory" in str(exc).lower():
            message = "Not enough memory. Try a smaller image and close other applications."
        elif isinstance(exc, (FileNotFoundError, ValueError, KeyError)):
            message = "The image or model could not be loaded. Check the model bundle and retry."
        else:
            message = "Enhancement failed. Retry the image; details are in the local application log."
        _write(status, {"state": "failed", "token": token, "message": message})


@dataclass
class Session:
    token: str
    valid: object = None
    name: str = "image"
    state: str = "idle"
    message: str = "Select an RGB JPEG or PNG to begin."
    process: object = None
    directory: Path | None = None
    started: float = 0
    touched: float = 0
    report: dict | None = None


class WorkerManager:
    def __init__(self, bundle_path=None, *, root=None, max_pixels=APP_MAX_PIXELS, idle_seconds=60):
        self.bundle = str(Path(bundle_path).resolve()) if bundle_path else None
        self.root = Path(root or Path(__file__).resolve().parents[2] / "artifacts/ui_jobs")
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_pixels, self.idle_seconds = max_pixels, idle_seconds
        self.sessions, self.lock = {}, threading.RLock()
        self.closed = threading.Event()
        self.janitor = threading.Thread(target=self._reap, daemon=True)
        self.janitor.start()
        atexit.register(self.close)

    def _stop(self, session):
        process = session.process
        if process:
            if process.is_alive():
                process.terminate()
                process.join(.3)
                if process.is_alive():
                    process.kill()
                    process.join(1)
            else:
                process.join()
            if process.is_alive():
                raise RuntimeError("Inference worker could not be stopped.")
            process.close()
            session.process = None

    def _clear(self, session):
        self._stop(session)
        if session.directory:
            shutil.rmtree(session.directory, ignore_errors=True)
            session.directory = None

    def _reap(self):
        while not self.closed.wait(5):
            self.reap_expired()

    def reap_expired(self):
        with self.lock:
            for key, session in list(self.sessions.items()):
                if time.monotonic() - session.touched > self.idle_seconds:
                    self._clear(session)
                    del self.sessions[key]

    def upload(self, key, path):
        with self.lock:
            if key in self.sessions:
                self._clear(self.sessions[key])
            session = Session(uuid.uuid4().hex, touched=time.monotonic())
            self.sessions[key] = session
            if path:
                try:
                    session.valid = validate_image(path, max_pixels=self.max_pixels)
                    w, h = session.valid.image.size
                    padded_pixels = ((w+7)//8*8) * ((h+7)//8*8)
                    if padded_pixels > int(self.max_pixels * 1.05) + 64:
                        raise ValueError("This image's dimensions require too much padding. Use an image with a less extreme aspect ratio.")
                    # Enforce development isolation before showing a held-out preview.
                    if self.bundle:
                        from .manifest import read_manifest
                        meta = json.loads(Path(self.bundle).read_text())
                        manifest = read_manifest(meta["configuration"]["manifest"])
                        sealed = [r for r in manifest["pairs"] if r["split"] == "test"] + manifest.get("challenging", [])
                        if session.valid.source_sha256 in {r["raw_sha256"] for r in sealed}:
                            raise ValueError("Final test and challenging inputs remain sealed during development.")
                    session.name = re.sub(r"[^\w.-]+", "_", Path(path).stem)[:100] or "image"
                    session.state = "ready"
                    w, h = session.valid.image.size
                    session.message = f"Ready · {w} × {h} pixels. Select Enhance to process this image."
                except (ValueError, OSError, KeyError) as exc:
                    session.valid = None
                    session.state = "failed"
                    session.message = f"Cannot preview image: {exc}"
            return session

    def start(self, key, token):
        with self.lock:
            session = self.sessions.get(key)
            if not session or session.token != token:
                return {"state": "stale"}
            session.touched = time.monotonic()
            if session.valid is None:
                session.state, session.message = "failed", "Select a valid image before enhancing."
                return self.poll(key, token)
            if not self.bundle:
                session.state, session.message = "failed", "No trained model is available. Start the app with a model bundle."
                return self.poll(key, token)
            if session.process and session.process.is_alive():
                return self.poll(key, token)
            if any(s.process and s.process.is_alive() for s in self.sessions.values()):
                return {"state": "busy", "message": "An image is already processing. Wait or cancel before retrying."}
            self._clear(session)
            session.directory = Path(tempfile.mkdtemp(prefix="job-", dir=self.root))
            # Save only sanitized RGB; source EXIF/GPS is never copied into job files.
            session.valid.image.save(session.directory / "input.png", format="PNG")
            session.started = time.monotonic()
            session.state, session.message, session.report = "loading", "Loading the enhancement model…", None
            process = mp.get_context("spawn").Process(target=_run, args=(
                str(session.directory), self.bundle, token, session.valid.source_sha256,
                session.name + "_enhanced.png", self.max_pixels), daemon=True)
            session.process = process
            try:
                process.start()
            except Exception:
                session.process = None
                self._clear(session)
                session.state, session.message = "failed", "Could not start processing. Retry the image."
                logging.exception("Could not spawn RGB worker")
            return self.poll(key, token)

    def poll(self, key, token):
        with self.lock:
            session = self.sessions.get(key)
            if not session:
                return {"state": "expired" if token else "stale", "token": token,
                        "message": "This session expired. Upload the image again.", "output": None, "valid_input": False}
            if token != session.token:
                return {"state": "stale"}
            session.touched = time.monotonic()
            result_path = session.directory / "status.json" if session.directory else None
            if session.process and session.state not in {"success", "failed", "cancelled"}:
                if time.monotonic() - session.started > 300:
                    self._clear(session)
                    session.state, session.message = "failed", "Processing took too long. Try a smaller image."
                elif result_path and result_path.exists():
                    record = json.loads(result_path.read_text())
                    if record.get("token") == session.token:
                        session.state = record["state"]
                        session.message = {"loading": "Loading the enhancement model…", "depth": "Estimating relative depth…", "processing": "Enhancing at original resolution…",
                            "exporting": "Preparing your PNG…", "success": "Enhancement complete. Compare the images or download the PNG.",
                            "failed": record.get("message", "Enhancement failed. Please retry.")}[session.state]
                        if session.state == "success":
                            session.report = record["report"]
                            session.report["end_to_end_seconds"] = record["completed_monotonic"]-session.started
                if session.process and not session.process.is_alive():
                    if session.state not in {"success", "failed"}:
                        # The child may have committed its result between the first
                        # file check and is_alive(). Read the terminal record again.
                        if result_path and result_path.exists():
                            final = json.loads(result_path.read_text())
                            if final.get("token") == token and final.get("state") in {"success", "failed"}:
                                session.state = final["state"]
                                session.message = final.get("message", "Enhancement complete. Compare the images or download the PNG.")
                                session.report = final.get("report")
                                if session.report:
                                    session.report["end_to_end_seconds"] = final["completed_monotonic"]-session.started
                        if session.state not in {"success", "failed"}:
                            session.state, session.message = "failed", "Processing stopped unexpectedly. Please retry."
                    self._stop(session)
            output = None
            if session.state == "success" and session.directory:
                output = str(session.directory / (session.name + "_enhanced.png"))
                if not Path(output).is_file():
                    session.state, session.message, output = "failed", "The PNG is unavailable. Please enhance the image again.", None
            return {"state": session.state, "token": token, "message": session.message,
                    "output": output, "report": session.report, "valid_input": session.valid is not None}

    def cancel(self, key, token):
        with self.lock:
            session = self.sessions.get(key)
            if not session or session.token != token:
                return {"state": "stale"}
            self._clear(session)
            session.state, session.message, session.report = "cancelled", "Cancelled. You can retry or choose another image.", None
            session.touched = time.monotonic()
            return self.poll(key, token)

    def remove(self, key):
        with self.lock:
            if key in self.sessions:
                self._clear(self.sessions.pop(key))

    def close(self):
        self.closed.set()
        with self.lock:
            for session in self.sessions.values():
                self._clear(session)
            self.sessions.clear()
