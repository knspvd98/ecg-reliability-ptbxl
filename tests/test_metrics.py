from __future__ import annotations

import numpy as np

from ecg_reliability.evaluation.calibration_metrics import compute_calibration_report
from ecg_reliability.evaluation.metrics import compute_classification_metrics


def test_metrics_are_finite() -> None:
    y_true = np.array(
        [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1],
        ],
        dtype=np.int64,
    )
    probs = np.array(
        [
            [0.9, 0.1, 0.1, 0.2, 0.2],
            [0.1, 0.8, 0.1, 0.2, 0.3],
            [0.2, 0.2, 0.75, 0.1, 0.1],
            [0.2, 0.3, 0.2, 0.8, 0.7],
        ],
        dtype=np.float32,
    )
    metrics = compute_classification_metrics(y_true, probs)
    calibration = compute_calibration_report(y_true, probs)
    assert np.isfinite(metrics["macro_auroc"])
    assert np.isfinite(metrics["macro_auprc"])
    assert np.isfinite(metrics["macro_f1"])
    assert np.isfinite(calibration["macro_ece"])
    assert np.isfinite(calibration["macro_brier"])
