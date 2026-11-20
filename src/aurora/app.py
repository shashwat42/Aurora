"""Local RGB prototype: interruptible enhancement, comparison and PNG export."""
import json
from pathlib import Path

from .image import validate_image, ImageValidationError
from .worker import WorkerManager


def configure_dependency_warnings():
    """Quiet the known Gradio/Starlette status alias on every queue poll."""
    import warnings
    from starlette.exceptions import StarletteDeprecationWarning
    warnings.filterwarnings(
        'ignore',
        message=r"^'HTTP_422_UNPROCESSABLE_ENTITY' is deprecated\. Use 'HTTP_422_UNPROCESSABLE_CONTENT' instead\.$",
        category=StarletteDeprecationWarning,
        module=r'^gradio\.routes$',
    )


def preview_image(path):
    if path is None:
        return None, "Select an RGB JPEG or PNG to begin."
    try:
        valid = validate_image(path)
        w, h = valid.image.size
        return valid.image, f"Ready for preview · {w} × {h} pixels · {valid.source_format}."
    except ImageValidationError as exc:
        return None, f"Cannot preview image: {exc}"


def default_bundle(root="."):
    root = Path(root)
    deployment = root / 'configs/experimental_model.json'
    if deployment.exists():
        from .manifest import file_hash, safe_path
        active = json.loads(deployment.read_text())
        record_path = safe_path(root, active['record'])
        if file_hash(record_path) != active['record_sha256']:
            raise ValueError('Experimental model decision identity changed.')
        record = json.loads(record_path.read_text())
        bundle_path = safe_path(root, record['bundle'])
        bundle = json.loads(bundle_path.read_text())
        if (record.get('status') != 'user_authorized_experimental_prototype'
                or file_hash(bundle_path) != record['bundle_sha256']
                or file_hash(safe_path(bundle_path.parent, bundle['weights'])) != record['weights_sha256']
                or bundle['weights_sha256'] != record['weights_sha256']):
            raise ValueError('Experimental model identity changed; restore the frozen artifact.')
        return str(bundle_path), False
    selected = root / "artifacts/phase2/evaluation/selection.json"
    ledger_path = root / "artifacts/phase2/evaluation/ledger.json"
    if selected.exists():
        record = json.loads(selected.read_text())
        if record.get("selected"):
            from .manifest import file_hash
            chosen = record["chosen"]
            if file_hash(chosen["bundle"]) != chosen["bundle_sha256"]:
                raise ValueError("Selected model identity changed; restore its original bundle.")
            return chosen["bundle"], True
        if record.get("status") == "no_artifact_eligible_candidate":
            return None, False
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text())
        chosen = next(c for c in ledger["candidates"] if c["name"] == ledger["provisional_metric_leader"])
        return chosen["bundle"], False
    last = root / "artifacts/phase2/rgb_baseline/last/bundle.json"
    return (str(last), False) if last.exists() else (None, False)


def create_app(bundle=None, *, preview_only=False):
    import os
    os.environ["GRADIO_TEMP_DIR"] = str(Path(__file__).resolve().parents[2] / "artifacts/gradio")
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    import gradio as gr
    configure_dependency_warnings()
    approved = False
    if preview_only:
        bundle = None
    elif bundle is None:
        bundle, approved = default_bundle()
    available = bundle is not None and Path(bundle).is_file()
    depth_enabled = available and bool(json.loads(Path(bundle).read_text()).get('configuration', {}).get('depth_config'))
    manager = WorkerManager(bundle if available else None)
    signatures = {}

    def key(request):
        return request.session_hash or "local"

    def render(record):
        if record["state"] == "stale":
            return [gr.skip()] * 6
        state = record["state"]
        active = state in {"loading", "depth", "processing", "exporting"}
        values = {"loading": .15, "depth": .3, "processing": .65, "exporting": .9, "success": 1}
        progress = f'<progress max="1" value="{values.get(state, 0)}" aria-label="Enhancement progress" style="width:100%"></progress>'
        image = None
        if record.get("output"):
            from PIL import Image
            with Image.open(record["output"]) as source:
                image = source.copy()
        message = record.get("message", "Select an image.")
        if state == "success" and record.get("report"):
            report = record["report"]
            message = f"Enhanced · {report['width']} × {report['height']} pixels. Compare the images or download the PNG."
        return [image, gr.update(value=record.get("output"), interactive=state == "success"),
                message, progress,
                gr.update(interactive=available and record.get("valid_input", False) and not active),
                gr.update(interactive=active)]

    def upload(path, request: gr.Request):
        session = manager.upload(key(request), path)
        signatures.pop(key(request), None)
        values = render(manager.poll(key(request), session.token))
        values[4] = gr.update(interactive=available and session.valid is not None)
        return [session.valid.image if session.valid else None, *values, session.token]

    def start(token, request: gr.Request):
        signatures.pop(key(request), None)
        return render(manager.start(key(request), token))

    def cancel(token, request: gr.Request):
        signatures.pop(key(request), None)
        return render(manager.cancel(key(request), token))

    def poll(token, request: gr.Request):
        record = manager.poll(key(request), token)
        signature = (record.get("token"), record["state"], record.get("message"), record.get("output"))
        if signatures.get(key(request)) == signature:
            return [gr.skip()] * 6
        signatures[key(request)] = signature
        return render(record)

    def unload(request: gr.Request):
        signatures.pop(key(request), None)
        manager.remove(key(request))

    with gr.Blocks(title="Aurora · RGB prototype", analytics_enabled=False, delete_cache=(60, 60)) as app:
        gr.Markdown('# Aurora\nUnderwater image enhancement · ' + ('Depth-guided prototype' if depth_enabled else 'RGB prototype'))
        if approved:
            gr.Markdown("Using the artifact-reviewed RGB baseline. Depth guidance comes in a later phase.")
        elif available:
            gr.Markdown(('Experimental depth-guided prototype' if depth_enabled else 'Experimental RGB prototype') + ' · Colour, highlight, texture and border defects may remain. Artifact approval has not been established.')
        else:
            gr.Markdown("Preview mode · A trained, eligible model is required to enable Enhance.")
        gr.Markdown("Choose a single-frame RGB JPEG or PNG, up to 20 MiB and 2.5 million pixels. The original resolution is preserved. Images stay on this computer; temporary processing files are removed after closing the page or 60 seconds of inactivity.")
        token = gr.State("")
        source = gr.File(label="Upload an underwater image", file_types=[".jpg", ".jpeg", ".png"], type="filepath")
        status = gr.Textbox(label="Status and dimensions", value="Select an RGB JPEG or PNG to begin.", interactive=False)
        with gr.Row():
            original = gr.Image(label="Original image", type="pil", interactive=False, format="png", height=360, buttons=["fullscreen"])
            enhanced = gr.Image(label="Enhanced image", type="pil", interactive=False, format="png", height=360, buttons=["fullscreen"])
        with gr.Row():
            enhance = gr.Button("Enhance", variant="primary", interactive=False)
            stop = gr.Button("Cancel", interactive=False)
            download = gr.DownloadButton("Download enhanced PNG", interactive=False)
        progress = gr.HTML('<progress max="1" value="0" aria-label="Enhancement progress" style="width:100%"></progress>')
        timer = gr.Timer(.5)
        outputs = [enhanced, download, status, progress, enhance, stop]
        poll_event = timer.tick(poll, inputs=token, outputs=outputs, queue=True,
                                concurrency_limit=1, api_name="status")
        source.change(upload, inputs=source, outputs=[original, *outputs, token], queue=False,
                      trigger_mode="always_last", cancels=[poll_event], api_name="upload")
        enhance.click(start, inputs=token, outputs=outputs, queue=False, api_name="enhance")
        stop.click(cancel, inputs=token, outputs=outputs, queue=False, api_name="cancel")
        app.unload(unload)
    app.aurora_manager = manager
    return app


def launch(port=7860, bundle=None, preview_only=False):
    create_app(bundle, preview_only=preview_only).queue().launch(
        server_name="127.0.0.1", server_port=port, share=False, inbrowser=False,
        max_file_size=20 * 1024 * 1024, footer_links=[])
