from __future__ import annotations

from collections.abc import Callable

import numpy as np


def bootstrap_metric(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 200,
    seed: int = 42,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    values = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        values.append(metric_fn(y_true[indices], y_score[indices]))
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p95": float(np.quantile(values, 0.95)),
    }
