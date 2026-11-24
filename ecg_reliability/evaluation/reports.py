from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ecg_reliability.data.labels import TARGET_LABELS


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)


def save_metrics_csv(path: str | Path, metrics: dict[str, Any]) -> None:
    rows = []
    for key, value in metrics.items():
        if key == "per_class":
            for label, label_metrics in value.items():
                for metric_name, metric_value in label_metrics.items():
                    rows.append({"scope": "per_class", "label": label, "metric": metric_name, "value": metric_value})
        elif isinstance(value, (int, float)):
            rows.append({"scope": "aggregate", "label": "all", "metric": key, "value": value})
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(target, index=False)


def save_prediction_bundle(
    output_dir: str | Path,
    split: str,
    ecg_ids: np.ndarray,
    logits: np.ndarray,
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> None:
    target = Path(output_dir) / split
    target.mkdir(parents=True, exist_ok=True)
    np.save(target / "ecg_ids.npy", ecg_ids.astype(np.int64))
    np.save(target / "logits.npy", logits.astype(np.float32))
    np.save(target / "probabilities.npy", probabilities.astype(np.float32))
    np.save(target / "labels.npy", labels.astype(np.float32))
    frame = pd.DataFrame(probabilities, columns=[f"prob_{label}" for label in TARGET_LABELS])
    for idx, label in enumerate(TARGET_LABELS):
        frame[f"logit_{label}"] = logits[:, idx]
        frame[f"target_{label}"] = labels[:, idx]
    frame.insert(0, "ecg_id", ecg_ids.astype(np.int64))
    frame.to_csv(target / "predictions.csv", index=False)
