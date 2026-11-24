from __future__ import annotations

from typing import Any

import numpy as np

from ecg_reliability.data.labels import TARGET_LABELS


def weight_arrays(decision_config: dict[str, Any], label_names: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    label_names = label_names or TARGET_LABELS
    weights = decision_config["harm_weights"]
    fn = np.asarray([weights[label]["fn"] for label in label_names], dtype=np.float32)
    fp = np.asarray([weights[label]["fp"] for label in label_names], dtype=np.float32)
    return fn, fp


def expected_harm_components(probabilities: np.ndarray, fn_weights: np.ndarray, fp_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    harm_positive = (1.0 - probabilities) * fp_weights
    harm_negative = probabilities * fn_weights
    return harm_positive, harm_negative


def minimal_expected_harm(probabilities: np.ndarray, fn_weights: np.ndarray, fp_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    harm_pos, harm_neg = expected_harm_components(probabilities, fn_weights, fp_weights)
    min_harm = np.minimum(harm_pos, harm_neg)
    decisions = (harm_pos < harm_neg).astype(int)
    return decisions, min_harm


def realized_harm(predictions: np.ndarray, labels: np.ndarray, fn_weights: np.ndarray, fp_weights: np.ndarray) -> np.ndarray:
    false_negatives = ((labels == 1) & (predictions == 0)).astype(np.float32) * fn_weights
    false_positives = ((labels == 0) & (predictions == 1)).astype(np.float32) * fp_weights
    return false_negatives.sum(axis=1) + false_positives.sum(axis=1)
