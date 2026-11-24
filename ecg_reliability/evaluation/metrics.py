from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, hamming_loss, roc_auc_score

from ecg_reliability.data.labels import ABNORMAL_LABELS, TARGET_LABELS


def _safe_metric(fn, y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        value = fn(y_true, y_score)
        return float(value)
    except ValueError:
        return float("nan")


def macro_ignore_nan(values: list[float]) -> float:
    valid = [value for value in values if not math.isnan(value)]
    return float(np.mean(valid)) if valid else float("nan")


def compute_classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray | None = None,
    label_names: list[str] | None = None,
) -> dict[str, Any]:
    label_names = label_names or TARGET_LABELS
    predictions = predictions if predictions is not None else (probabilities >= 0.5).astype(int)
    per_class: dict[str, dict[str, float]] = {}
    class_aurocs: list[float] = []
    class_auprcs: list[float] = []
    for idx, label in enumerate(label_names):
        auroc = _safe_metric(roc_auc_score, y_true[:, idx], probabilities[:, idx])
        auprc = _safe_metric(average_precision_score, y_true[:, idx], probabilities[:, idx])
        per_class[label] = {"auroc": auroc, "auprc": auprc}
        class_aurocs.append(auroc)
        class_auprcs.append(auprc)

    abnormal_indices = [label_names.index(label) for label in ABNORMAL_LABELS if label in label_names]
    abnormal_aurocs = [class_aurocs[idx] for idx in abnormal_indices]
    abnormal_auprcs = [class_auprcs[idx] for idx in abnormal_indices]
    return {
        "macro_auroc": macro_ignore_nan(class_aurocs),
        "macro_auprc": macro_ignore_nan(class_auprcs),
        "abnormal_macro_auroc": macro_ignore_nan(abnormal_aurocs),
        "abnormal_macro_auprc": macro_ignore_nan(abnormal_auprcs),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, predictions, average="micro", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_true, predictions)),
        "per_class": per_class,
    }
