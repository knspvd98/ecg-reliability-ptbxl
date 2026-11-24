from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    # This project stores trusted local training artifacts with config and history,
    # so checkpoint loading must allow more than bare tensor weights.
    return torch.load(Path(path), map_location=map_location, weights_only=False)
