from __future__ import annotations

import numpy as np

from ecg_reliability.data.labels import ABNORMAL_LABELS, TARGET_LABELS


def apply_norm_consistency_rule(predictions: np.ndarray, label_names: list[str] | None = None) -> np.ndarray:
    label_names = label_names or TARGET_LABELS
    adjusted = predictions.copy()
    norm_idx = label_names.index("NORM")
    abnormal_indices = [label_names.index(label) for label in ABNORMAL_LABELS if label in label_names]
    abnormal_active = adjusted[:, abnormal_indices].sum(axis=1) > 0
    adjusted[abnormal_active, norm_idx] = 0
    return adjusted


def norm_conflict_report(
    predictions_before: np.ndarray,
    predictions_after: np.ndarray,
    label_names: list[str] | None = None,
) -> dict[str, float | int]:
    label_names = label_names or TARGET_LABELS
    norm_idx = label_names.index("NORM")
    abnormal_indices = [label_names.index(label) for label in ABNORMAL_LABELS if label in label_names]
    raw_conflicts = ((predictions_before[:, norm_idx] == 1) & (predictions_before[:, abnormal_indices].sum(axis=1) > 0))
    post_conflicts = ((predictions_after[:, norm_idx] == 1) & (predictions_after[:, abnormal_indices].sum(axis=1) > 0))
    changed_norm = predictions_before[:, norm_idx] != predictions_after[:, norm_idx]
    return {
        "raw_conflict_count": int(raw_conflicts.sum()),
        "raw_conflict_rate": float(raw_conflicts.mean()),
        "post_conflict_count": int(post_conflicts.sum()),
        "post_conflict_rate": float(post_conflicts.mean()),
        "n_changed_norm_decisions": int(changed_norm.sum()),
    }
