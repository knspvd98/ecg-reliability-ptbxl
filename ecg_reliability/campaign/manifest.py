from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from ecg_reliability.evaluation.reports import save_json


def build_campaign_manifest(**kwargs: Any) -> dict[str, Any]:
    return {
        "created_at": datetime.now().isoformat(),
        **kwargs,
    }


def write_campaign_manifest(path: str | Path, payload: dict[str, Any]) -> None:
    save_json(path, payload)
