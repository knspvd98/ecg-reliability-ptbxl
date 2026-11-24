from __future__ import annotations

import json
import os
import platform
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIRS = [
    "index",
    "cache",
    "checkpoints",
    "predictions",
    "calibration",
    "decision",
    "robustness",
    "explanations",
    "external",
    "figures",
    "tables",
    "summary",
    "campaign",
]


def project_root() -> Path:
    return PROJECT_ROOT


def resolve_path(path_str: str | os.PathLike[str]) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def ensure_output_structure() -> None:
    (PROJECT_ROOT / "outputs").mkdir(exist_ok=True)
    for name in OUTPUT_DIRS:
        (PROJECT_ROOT / "outputs" / name).mkdir(parents=True, exist_ok=True)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = resolve_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def snapshot_environment(path: str | Path) -> None:
    payload = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cuda_available": bool(torch and torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch else 0,
        "torch_version": getattr(torch, "__version__", None),
    }
    write_json(path, payload)
