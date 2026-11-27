from __future__ import annotations

from typing import Any

import numpy as np


def _bin_edges(probabilities: np.ndarray, num_bins: int, adaptive: bool = False) -> np.ndarray:
    if adaptive:
        return np.quantile(probabilities, np.linspace(0.0, 1.0, num_bins + 1))
    return np.linspace(0.0, 1.0, num_bins + 1)


def classwise_ece(y_true: np.ndarray, probabilities: np.ndarray, num_bins: int = 15, adaptive: bool = False) -> np.ndarray:
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)
    if y_true.ndim != 2 or y_true.shape != probabilities.shape or not y_true.size:
        raise ValueError("Expected matching, non-empty sample-by-label arrays")
    if num_bins < 1 or not np.isin(y_true, [0, 1]).all():
        raise ValueError("Positive bin count and binary targets are required")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Expected finite probabilities in [0, 1]")
    eces = []
    for idx in range(y_true.shape[1]):
        y = y_true[:, idx]
        p = probabilities[:, idx]
        edges = _bin_edges(p, num_bins=num_bins, adaptive=adaptive)
        # Assign ties to exactly one bin, including a repeated maximum edge.
        bins = np.searchsorted(edges[1:-1], p, side="right")
        residual_sums = np.bincount(bins, weights=y - p, minlength=num_bins)
        eces.append(np.abs(residual_sums).sum() / p.size)
    return np.asarray(eces, dtype=np.float32)


def brier_score(y_true: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    residuals = np.asarray(probabilities, dtype=float) - y_true
    return np.mean(residuals ** 2, axis=0)


def reliability_bins(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_index: int,
    num_bins: int = 15,
    *,
    adaptive: bool = False,
) -> list[dict[str, float]]:
    y = y_true[:, class_index]
    p = probabilities[:, class_index]
    edges = _bin_edges(p, num_bins=num_bins, adaptive=adaptive)
    bin_ids = np.searchsorted(edges[1:-1], p, side="right")
    bins: list[dict[str, float]] = []
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        mask = (bin_ids == index) & (p >= edges[0]) & (p <= edges[-1])
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
