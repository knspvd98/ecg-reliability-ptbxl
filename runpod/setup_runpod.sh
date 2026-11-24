#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python - <<'PY'
import json
import os
import platform
try:
    import torch
except Exception:
    torch = None
payload = {
    "platform": platform.platform(),
    "python": platform.python_version(),
    "ptbxl_root": os.environ.get("PTBXL_ROOT"),
    "torch": getattr(torch, "__version__", None),
    "cuda_available": bool(torch and torch.cuda.is_available()),
}
print(json.dumps(payload, indent=2))
PY
