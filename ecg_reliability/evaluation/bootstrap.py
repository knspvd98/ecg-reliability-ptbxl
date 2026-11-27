from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from typing import Any

import numpy as np


def _build_cluster_index_map(cluster_ids: np.ndarray) -> tuple[list[np.ndarray], int]:
    ids = np.asarray(cluster_ids)
    unique_clusters, inverse = np.unique(ids, return_inverse=True)
    groups: list[np.ndarray] = [np.flatnonzero(inverse == i) for i in range(len(unique_clusters))]
    return groups, len(unique_clusters)


def _sample_indices(
    rng: np.random.Generator,
    n_records: int,
    cluster_groups: list[np.ndarray] | None,
) -> np.ndarray:
    if cluster_groups is None:
        return rng.integers(0, n_records, size=n_records)
    chosen_clusters = rng.integers(0, len(cluster_groups), size=len(cluster_groups))
    return np.concatenate([cluster_groups[int(idx)] for idx in chosen_clusters], axis=0)


def _safe_eval(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> tuple[float | None, str | None]:
    try:
        value = float(metric_fn(y_true, y_score))
    except Exception as exc:
        return None, f"{exc.__class__.__name__}: {exc}"
    if not math.isfinite(value):
        return None, "non_finite_metric"
    return value, None


def bootstrap_metric(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 200,
    seed: int = 42,
    *,
    cluster_ids: np.ndarray | None = None,
    ci_level: float = 0.95,
) -> dict[str, Any]:
    """Compute percentile bootstrap confidence intervals with optional patient-cluster resampling.

    Tracks invalid/non-finite bootstrap resamples explicitly rather than silently
    discarding them, and reports confidence interval definition, resampling unit,
    and effective sample size.
    """
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score)
    if len(y_true_arr) != len(y_score_arr):
        raise ValueError(f"Mismatched lengths: y_true={len(y_true_arr)}, y_score={len(y_score_arr)}")
    if len(y_true_arr) == 0:
        raise ValueError("Cannot bootstrap an empty evaluation array.")
    if not (0.0 < ci_level < 1.0):
        raise ValueError(f"ci_level must lie in (0, 1), got {ci_level}")
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap}")

    n_records = int(len(y_true_arr))
    cluster_groups: list[np.ndarray] | None = None
    resampling_unit = "record_iid"
    n_clusters = n_records
    if cluster_ids is not None:
        if len(cluster_ids) != n_records:
            raise ValueError(f"cluster_ids length ({len(cluster_ids)}) does not match y_true ({n_records})")
        cluster_groups, n_clusters = _build_cluster_index_map(cluster_ids)
        resampling_unit = "patient_cluster"

    point_estimate, point_error = _safe_eval(metric_fn, y_true_arr, y_score_arr)
    rng = np.random.default_rng(seed)
    valid_values: list[float] = []
    invalid_reasons: Counter[str] = Counter()

    for _ in range(int(n_bootstrap)):
        indices = _sample_indices(rng, n_records, cluster_groups)
        value, err = _safe_eval(metric_fn, y_true_arr[indices], y_score_arr[indices])
        if err is not None or value is None:
            invalid_reasons[err or "unknown_error"] += 1
        else:
            valid_values.append(value)

    alpha = 1.0 - float(ci_level)
    low_q = alpha / 2.0
    high_q = 1.0 - alpha / 2.0
    ci_pct = int(round(ci_level * 100))
    ci_definition = f"percentile_{ci_pct}pct_{resampling_unit}"

    n_valid = len(valid_values)
    n_invalid = int(n_bootstrap) - n_valid
    if n_valid == 0:
        mean_val = float("nan")
        std_val = float("nan")
        ci_low = float("nan")
        ci_high = float("nan")
        p05 = float("nan")
        p95 = float("nan")
        valid_ci = False
    else:
        arr = np.asarray(valid_values, dtype=np.float64)
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr, ddof=1)) if n_valid > 1 else 0.0
        ci_low = float(np.quantile(arr, low_q))
        ci_high = float(np.quantile(arr, high_q))
        p05 = float(np.quantile(arr, 0.05))
        p95 = float(np.quantile(arr, 0.95))
        valid_ci = True

    return {
        "point_estimate": float(point_estimate) if point_estimate is not None else float("nan"),
        "point_estimate_error": point_error,
        "mean": mean_val,
        "std": std_val,
        "ci_level": float(ci_level),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p05": p05,
        "p95": p95,
        "ci_definition": ci_definition,
        "resampling_unit": resampling_unit,
        "n_records": n_records,
        "n_clusters": int(n_clusters),
        "effective_sample_size": int(n_clusters if cluster_groups is not None else n_records),
        "n_bootstrap_requested": int(n_bootstrap),
        "n_bootstrap_valid": int(n_valid),
        "n_bootstrap_invalid": int(n_invalid),
        "invalid_resample_rate": float(n_invalid / max(int(n_bootstrap), 1)),
        "invalid_reasons": dict(invalid_reasons),
        "valid_ci": valid_ci,
    }


def paired_bootstrap_comparison(
    y_true: np.ndarray,
    y_score_a: np.ndarray,
    y_score_b: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_bootstrap: int = 200,
    seed: int = 42,
    cluster_ids: np.ndarray | None = None,
    ci_level: float = 0.95,
) -> dict[str, Any]:
    """Paired bootstrap comparison (Model A minus Model B) on identical resampled units.

    Uses the exact same record or patient-cluster indices for both models on every
    resample to eliminate Monte Carlo comparison variance.
    """
    y_true_arr = np.asarray(y_true)
    score_a = np.asarray(y_score_a)
    score_b = np.asarray(y_score_b)
    if not (len(y_true_arr) == len(score_a) == len(score_b)):
        raise ValueError("y_true, y_score_a, and y_score_b must have identical length.")
    if len(y_true_arr) == 0:
        raise ValueError("Cannot run paired bootstrap comparison on empty arrays.")

    n_records = int(len(y_true_arr))
    cluster_groups: list[np.ndarray] | None = None
    resampling_unit = "record_iid"
    n_clusters = n_records
    if cluster_ids is not None:
        if len(cluster_ids) != n_records:
            raise ValueError(f"cluster_ids length ({len(cluster_ids)}) does not match y_true ({n_records})")
        cluster_groups, n_clusters = _build_cluster_index_map(cluster_ids)
        resampling_unit = "patient_cluster"

    point_a, err_a = _safe_eval(metric_fn, y_true_arr, score_a)
    point_b, err_b = _safe_eval(metric_fn, y_true_arr, score_b)
    delta_point = (point_a - point_b) if (point_a is not None and point_b is not None) else float("nan")

    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    invalid_reasons: Counter[str] = Counter()

    for _ in range(int(n_bootstrap)):
        indices = _sample_indices(rng, n_records, cluster_groups)
        val_a, e_a = _safe_eval(metric_fn, y_true_arr[indices], score_a[indices])
        val_b, e_b = _safe_eval(metric_fn, y_true_arr[indices], score_b[indices])
        if e_a is not None or e_b is not None or val_a is None or val_b is None:
            invalid_reasons[e_a or e_b or "unknown_error"] += 1
        else:
            deltas.append(val_a - val_b)

    alpha = 1.0 - float(ci_level)
    low_q = alpha / 2.0
    high_q = 1.0 - alpha / 2.0
    ci_pct = int(round(ci_level * 100))
    ci_definition = f"paired_percentile_{ci_pct}pct_{resampling_unit}"

    n_valid = len(deltas)
    n_invalid = int(n_bootstrap) - n_valid
    if n_valid == 0:
        return {
            "point_a": float(point_a) if point_a is not None else float("nan"),
            "point_b": float(point_b) if point_b is not None else float("nan"),
            "point_error": err_a or err_b,
            "delta_point_estimate": float(delta_point),
            "delta_mean": float("nan"),
            "delta_std": float("nan"),
            "delta_ci_low": float("nan"),
            "delta_ci_high": float("nan"),
            "prob_a_better": float("nan"),
            "p_value_two_sided": float("nan"),
            "ci_level": float(ci_level),
            "ci_definition": ci_definition,
            "resampling_unit": resampling_unit,
            "n_records": n_records,
            "n_clusters": int(n_clusters),
            "effective_sample_size": int(n_clusters if cluster_groups is not None else n_records),
            "n_bootstrap_requested": int(n_bootstrap),
            "n_bootstrap_valid": 0,
            "n_bootstrap_invalid": int(n_invalid),
            "invalid_resample_rate": 1.0,
            "invalid_reasons": dict(invalid_reasons),
            "valid_ci": False,
        }

    arr = np.asarray(deltas, dtype=np.float64)
    prob_gt = float(np.mean(arr > 0.0))
    prob_lt = float(np.mean(arr < 0.0))
    # Two-sided empirical bootstrap p-value with +1 continuity correction
    tail = min(float(np.sum(arr <= 0.0)), float(np.sum(arr >= 0.0)))
    p_value = float(min(1.0, 2.0 * (tail + 1.0) / (n_valid + 1.0)))
    return {
        "point_a": float(point_a) if point_a is not None else float("nan"),
        "point_b": float(point_b) if point_b is not None else float("nan"),
        "point_error": err_a or err_b,
        "delta_point_estimate": float(delta_point),
        "delta_mean": float(np.mean(arr)),
        "delta_std": float(np.std(arr, ddof=1)) if n_valid > 1 else 0.0,
        "delta_ci_low": float(np.quantile(arr, low_q)),
        "delta_ci_high": float(np.quantile(arr, high_q)),
        "prob_a_better": prob_gt,
        "prob_b_better": prob_lt,
        "p_value_two_sided": p_value,
        "ci_level": float(ci_level),
        "ci_definition": ci_definition,
        "resampling_unit": resampling_unit,
        "n_records": n_records,
        "n_clusters": int(n_clusters),
        "effective_sample_size": int(n_clusters if cluster_groups is not None else n_records),
        "n_bootstrap_requested": int(n_bootstrap),
        "n_bootstrap_valid": int(n_valid),
        "n_bootstrap_invalid": int(n_invalid),
        "invalid_resample_rate": float(n_invalid / max(int(n_bootstrap), 1)),
        "invalid_reasons": dict(invalid_reasons),
        "valid_ci": True,
    }
