from __future__ import annotations

import numpy as np

from ecg_reliability.calibration.apply import apply_calibrator, fit_calibrator


def _synthetic_logits_and_labels():
    logits = np.array(
        [
            [2.0, -1.0, 0.5, -0.4, -0.2],
            [-1.0, 2.0, -0.2, 0.1, 0.2],
            [0.6, 0.4, 1.5, -1.0, -0.8],
            [-0.4, -0.6, -0.9, 1.3, 1.7],
            [1.2, -0.3, -0.5, 0.4, -0.1],
            [-0.9, 1.0, 0.7, -0.7, 0.9],
        ],
        dtype=np.float32,
    )
    labels = np.array(
        [
            [1, 0, 1, 0, 0],
            [0, 1, 0, 0, 1],
            [1, 0, 1, 0, 0],
            [0, 0, 0, 1, 1],
            [1, 0, 0, 1, 0],
            [0, 1, 1, 0, 1],
        ],
        dtype=np.float32,
    )
    return logits, labels


def test_calibration_methods_return_valid_probabilities() -> None:
    logits, labels = _synthetic_logits_and_labels()
    for method in ["scalar_temperature", "vector_temperature", "platt", "isotonic"]:
        calibrator = fit_calibrator(method, logits, labels)
        probs = apply_calibrator(calibrator, logits)
        assert probs.shape == logits.shape
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)


def test_platt_handles_constant_class() -> None:
    logits, labels = _synthetic_logits_and_labels()
    labels[:, 0] = 1.0
    calibrator = fit_calibrator("platt", logits, labels)
    probs = apply_calibrator(calibrator, logits)
    assert np.allclose(probs[:, 0], 1.0)


def test_adaptive_ece_assigns_repeated_edges_once():
    from ecg_reliability.evaluation.calibration_metrics import classwise_ece

    p = np.array([[0.8, 0], [0.8, 1], [0.8, 1], [0.8, 1]])
    y = np.array([[1, 0], [1, 1], [1, 1], [1, 0]])
    for bins in (1, 4, 15):
        np.testing.assert_allclose(classwise_ece(y, p, bins, adaptive=True), [0.2, 0.25])
