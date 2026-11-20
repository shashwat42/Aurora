"""Acquire the exact Small checkpoint locally; retain its card and license evidence."""
from pathlib import Path
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import HfApi, snapshot_download

repo = "depth-anything/Depth-Anything-V2-Small-hf"
config_path = ROOT / "configs/depth.json"
existing = json.loads(config_path.read_text()) if config_path.exists() else {}
revision = existing.get("revision") or HfApi().model_info(repo).sha
directory = ROOT / "artifacts/models/depth-anything-v2-small"
snapshot_download(repo, revision=revision, local_dir=directory,
                  allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors", "README.md", "LICENSE*"])
card = (directory / "README.md").read_text()
if "license: apache-2.0" not in card:
    raise SystemExit("The exact checkpoint card does not declare the expected Apache-2.0 license.")
files = {}
for name in ["config.json", "preprocessor_config.json", "model.safetensors", "README.md"]:
    files[name] = hashlib.sha256((directory / name).read_bytes()).hexdigest()
record = {
    "model_id": repo, "revision": revision,
    "local_directory": str(directory.relative_to(ROOT)), "device": "cpu", "threads": 4,
    "weight_format": "safetensors", "files_sha256": files,
    "license": "Apache-2.0", "license_evidence": f"https://huggingface.co/{repo}/blob/{revision}/README.md",
    "license_scope": "This exact Small-hf checkpoint; do not infer terms for Base/Large variants.",
    "redistribution": "Apache-2.0 conditions apply, including license and notices; weights remain local and are not bundled in source delivery.",
    "preprocessing_version": "depth-srgb-518-aspect14-bilinear-minmax-v1",
    "normalization": "Per-image (relative - minimum) / (maximum - minimum); constant maps become zeros.",
    "resize": "Official DPTImageProcessor, slow/PIL path; preserve aspect, ensure multiple of 14, no cropping/padding; restore bilinear align_corners=False to oriented native RGB before normalization.",
    "processor_configuration": json.loads((directory / "preprocessor_config.json").read_text()),
}
config_path.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
