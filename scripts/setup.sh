#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export UV_CACHE_DIR="$PWD/.cache/uv"
export PIP_CACHE_DIR="$PWD/.cache/pip"
aurora_python="${AURORA_PYTHON:-python3}"
"$aurora_python" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 14) else "Aurora requires Python 3.14. Set AURORA_PYTHON to its executable.")'
if [[ ! -x .venv/bin/python ]]; then
    "$aurora_python" -m venv .venv
fi
.venv/bin/python -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 14) else "Existing .venv uses another Python version. Recreate it with Python 3.14 before setup.")'
if [[ ! -x .venv/bin/uv ]]; then
    .venv/bin/python -m pip install 'uv==0.13.6'
fi
.venv/bin/uv pip sync --python .venv/bin/python --index-strategy unsafe-best-match requirements.lock.txt
.venv/bin/uv pip check --python .venv/bin/python
.venv/bin/python -m aurora --help
