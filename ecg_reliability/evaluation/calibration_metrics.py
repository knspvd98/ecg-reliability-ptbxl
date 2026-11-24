from __future__ import annotations

from typing import Any

import numpy as np


def _bin_edges(probabilities: np.ndarray, num_bins: int, adaptive: bool = False) -> np.ndarray:
    if adaptive:
        return np.quantile(probabilities, np.linspace(0.0, 1.0, num_bins + 1))
    return np.linspace(0.0, 1.0, num_bins + 1)


def classwise_ece(y_true: np.ndarray, probabilities: np.ndarray, num_bins: int = 15, adaptive: bool = False) -> np.ndarray:
    eces = []
    for idx in range(y_true.shape[1]):
        y = y_true[:, idx]
        p = probabilities[:, idx]
        edges = _bin_edges(p, num_bins=num_bins, adaptive=adaptive)
        ece = 0.0
        for left, right in zip(edges[:-1], edges[1:], strict=True):
            mask = (p >= left) & (p <= right) if right == edges[-1] else (p >= left) & (p < right)
            if not np.any(mask):
                continue
            acc = y[mask].mean()
            conf = p[mask].mean()
            ece += abs(acc - conf) * (mask.mean())
        eces.append(float(ece))
    return np.asarray(eces, dtype=np.float32)


def brier_score(y_true: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    return np.mean((probabilities - y_true) ** 2, axis=0)


def reliability_bins(y_true: np.ndarray, probabilities: np.ndarray, class_index: int, num_bins: int = 15) -> list[dict[str, float]]:
    y = y_true[:, class_index]
    p = probabilities[:, class_index]
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    bins: list[dict[str, float]] = []
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        mask = (p >= left) & (p <= right if right == edges[-1] else p < right)
        if not np.any(mask):
            bins.append({"bin_left": float(left), "bin_right": float(right), "accuracy": float("nan"), "confidence": float("nan"), "count": 0})
            continue
        bins.append(
            {
                "bin_left": float(left),
                "bin_right": float(right),
                "accuracy": float(y[mask].mean()),
                "confidence": float(p[mask].mean()),
                "count": int(mask.sum()),
            }
        )
    return bins


def compute_calibration_report(y_true: np.ndarray, probabilities: np.ndarray, num_bins: int = 15) -> dict[str, Any]:
    ece = classwise_ece(y_true, probabilities, num_bins=num_bins, adaptive=False)
    adaptive = classwise_ece(y_true, probabilities, num_bins=num_bins, adaptive=True)
    brier = brier_score(y_true, probabilities)
    return {
        "ece_per_class": ece.tolist(),
        "adaptive_ece_per_class": adaptive.tolist(),
        "brier_per_class": brier.tolist(),
        "macro_ece": float(np.nanmean(ece)),
        "macro_adaptive_ece": float(np.nanmean(adaptive)),
        "macro_brier": float(np.nanmean(brier)),
    }
