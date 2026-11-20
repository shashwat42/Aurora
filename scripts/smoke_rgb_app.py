"""Short live-server check with two diagnostic images; no selection/test scoring."""
import json
import argparse
from pathlib import Path
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def downloaded(value):
    if isinstance(value, str) and Path(value).is_file():
        return value
    if isinstance(value, dict):
        for key in ["value", "path"]:
            if key in value:
                found = downloaded(value[key])
                if found:
                    return found
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", help="Explicit experimental bundle; otherwise use the app's current default/selected model")
    parser.add_argument('--output', help='New report path; refuses overwriting existing evidence')
    args = parser.parse_args()
    report_path = Path(args.output) if args.output else None
    if report_path:
        if report_path.exists():
            raise FileExistsError(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
    from gradio_client import Client, handle_file
    from PIL import Image
    from aurora.app import create_app, default_bundle
    from aurora.image import validate_image
    from aurora.manifest import read_manifest, file_hash
    resolved, approved = (args.bundle, False) if args.bundle else default_bundle()
    if not resolved:
        raise ValueError("No eligible default model is available for the UI check.")
    bundle = Path(resolved).resolve()
    depth_expected = bool(json.loads(bundle.read_text())['configuration'].get('depth_config'))
    manifest = read_manifest(ROOT / "data/metadata/manifest.v1.json")
    diagnostics = {r["id"]: r for r in manifest["pairs"] if r["split"] == "diagnostic"}
    chosen = [diagnostics["14_img_"], max(diagnostics.values(), key=lambda r: r["width"]*r["height"])]
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    app = create_app(str(bundle)) if args.bundle else create_app()
    records = []
    try:
        app.queue().launch(server_name="127.0.0.1", server_port=port, share=False,
                           prevent_thread_lock=True, quiet=True, footer_links=[])
        client = Client(f"http://127.0.0.1:{port}", verbose=False, analytics_enabled=False,
                        download_files=ROOT / "artifacts/phase2/ui_downloads")
        for row in chosen:
            path = ROOT / "data" / row["raw_path"]
            uploaded = client.predict(handle_file(str(path)), api_name="/upload")
            assert "Ready" in uploaded[3], uploaded
            assert uploaded[1] is None and downloaded(uploaded[2]) is None
            client.predict(api_name="/enhance")
            deadline = time.monotonic()+30
            while time.monotonic() < deadline:
                status = client.predict(api_name="/status")
                if isinstance(status[2], str) and status[2].startswith("Enhanced"):
                    break
                if isinstance(status[2], str) and ("failed" in status[2].lower() or "could not" in status[2].lower()):
                    raise AssertionError(status)
                time.sleep(.1)
            else:
                raise AssertionError("Live inference did not finish within the short-check timeout")
            output = downloaded(status[1])
            assert output, status
            valid = validate_image(path)
            with Image.open(output) as png:
                assert png.size == valid.image.size and png.mode == "RGB"
                assert png.info.get("icc_profile") and not png.getexif()
                changed = png.tobytes() != valid.image.tobytes()
            session = app.aurora_manager.sessions[client.session_hash]
            if depth_expected:assert session.report['depth_mode'] == 'live'
            records.append({"id": row["id"], "width": valid.image.width, "height": valid.image.height,
                            "download_reopened": True, "source_metadata_stripped": True, "srgb_profile": True,
                            "output_pixels_changed": changed, **session.report})
        # Cancel a real pending operation through the UI endpoints, then verify clearing.
        client.predict(api_name="/enhance")
        cancelled_phase = 'pending'
        if depth_expected:
            deadline=time.monotonic()+30
            while time.monotonic()<deadline:
                state=client.predict(api_name='/status')
                session=app.aurora_manager.sessions[client.session_hash]
                if session.state=='depth':
                    cancelled_phase='depth';break
                if session.state in {'failed','success'}:raise AssertionError('Did not observe live depth stage before cancellation')
                time.sleep(.05)
            else:raise AssertionError('Live depth stage not reached')
        before = time.perf_counter()
        cancelled = client.predict(api_name="/cancel")
        cancel_seconds = time.perf_counter()-before
        assert "Cancelled" in cancelled[2] and downloaded(cancelled[1]) is None
        assert app.aurora_manager.sessions[client.session_hash].process is None
        client.predict(api_name='/enhance')
        deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            retried=client.predict(api_name='/status')
            if isinstance(retried[2],str) and retried[2].startswith('Enhanced'):break
            if app.aurora_manager.sessions[client.session_hash].state=='failed':raise AssertionError(retried)
            time.sleep(.1)
        else:raise AssertionError('Retry after cancel did not complete')
        assert downloaded(retried[1])
        corrupt = (report_path.parent if report_path else ROOT / 'artifacts/phase2') / 'corrupt-upload.png'
        corrupt.write_bytes(b"not an image")
        invalid = client.predict(handle_file(str(corrupt)), api_name="/upload")
        assert invalid[0] is None and invalid[1] is None and downloaded(invalid[2]) is None
        assert "Cannot preview" in invalid[3]
        after_invalid = client.predict(api_name="/status")
        assert "Cannot preview" in after_invalid[2], after_invalid
        session = app.aurora_manager.sessions[client.session_hash]
        assert not app.aurora_manager.poll(client.session_hash, session.token)["valid_input"]
        client.close()
        report = {"status": "passed", "mode": 'experimental depth-guided MVP' if depth_expected else 'experimental RGB prototype',
                  "weights_sha256": json.loads(bundle.read_text())["weights_sha256"],
                  "manifest_sha256": file_hash(ROOT / "data/metadata/manifest.v1.json"),
                  "two_successive_diagnostic_uploads": records, "cancel_rpc_seconds": cancel_seconds,
                  "cancel_worker_released": True, "invalid_upload_clears_results": True,
                  "cancelled_phase": cancelled_phase, 'retry_after_cancel_passed':True,
                  "selection_images_used": 0, "test_images_scored": 0, "challenging_images_scored": 0,
                  "note": "API-driven live UI workflow check, not a manual browser usability/artifact review or Phase 3 performance contract."}
        report_name = "app_smoke_selected.json" if approved else "app_smoke.json"
        destination = report_path or ROOT / 'artifacts/phase2' / report_name
        with destination.open('x' if report_path else 'w') as stream:
            stream.write(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2))
    finally:
        app.aurora_manager.close()
        app.close()


if __name__ == "__main__":
    main()
