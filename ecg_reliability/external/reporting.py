from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from ecg_reliability.evaluation.calibration_metrics import compute_calibration_report


def safe_metric(fn, y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(fn(y_true, y_score))
    except ValueError:
        return float("nan")


def compute_binary_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float32).reshape(-1)
    calibration = compute_calibration_report(y_true[:, None], y_score[:, None])
    return {
        "auroc": safe_metric(roc_auc_score, y_true, y_score),
        "auprc": safe_metric(average_precision_score, y_true, y_score),
        "brier": float(np.mean((y_score - y_true) ** 2)),
        "ece": float(calibration["macro_ece"]),
        "n_records": int(len(y_true)),
        "positive_rate": float(y_true.mean()) if len(y_true) else float("nan"),
    }


def compute_sourcewise_binary_metrics(frame: pd.DataFrame, label_column: str, score_column: str) -> pd.DataFrame:
    rows = []
    for source, subset in frame.groupby("source"):
        metrics = compute_binary_metrics(
            subset[label_column].to_numpy(dtype=np.float32),
            subset[score_column].to_numpy(dtype=np.float32),
        )
        rows.append({"source": source, **metrics})
    return pd.DataFrame(rows)


def load_snomed_overlap_mapping(path: str | Path) -> dict[str, Any] | None:
    mapping_path = Path(path)
    if not mapping_path.exists():
        return None
    with mapping_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return payload if payload else None
