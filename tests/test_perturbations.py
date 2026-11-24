from __future__ import annotations

import numpy as np

from ecg_reliability.robustness.evaluate_robustness import (
    decision_layer_unavailable_payload,
    evaluate_decision_layer_under_perturbations,
    evaluate_under_perturbations,
)
from ecg_reliability.robustness.perturbations import (
    amplitude_scaling,
    baseline_wander,
    gaussian_noise,
    lead_dropout_group,
    lead_dropout_single,
)


def test_perturbations_preserve_shape_and_effect() -> None:
    signal = np.ones((12, 1000), dtype=np.float32)
    dropped_single = lead_dropout_single(signal, lead_index=3)
    dropped_group = lead_dropout_group(signal, group="limb")
    noisy = gaussian_noise(signal, std=0.01, rng=np.random.default_rng(0))
    wandered = baseline_wander(signal, amplitude=0.05, frequency_hz=0.33, sampling_rate=100)
    scaled = amplitude_scaling(signal, scale=1.1)

    for arr in [dropped_single, dropped_group, noisy, wandered, scaled]:
        assert arr.shape == signal.shape

    assert np.allclose(dropped_single[3], 0.0)
    assert np.allclose(dropped_group[:6], 0.0)
    assert not np.allclose(noisy, signal)
    assert not np.allclose(wandered, signal)
    assert np.allclose(scaled, signal * 1.1)


def test_robustness_results_include_delta_summary() -> None:
    signals = np.random.randn(4, 12, 1000).astype(np.float32)
    labels = np.array(
        [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1],
        ],
        dtype=np.int64,
    )

    def predict_proba_fn(batch: np.ndarray) -> np.ndarray:
        base = np.tile(np.array([[0.7, 0.2, 0.2, 0.2, 0.2]], dtype=np.float32), (len(batch), 1))
        return base

    results = evaluate_under_perturbations(signals, labels, predict_proba_fn, {"gaussian_noise": {"std": 0.01}})
    assert "delta_summary" in results
    assert len(results["delta_summary"]) == 1
    assert "delta_macro_auroc" in results["gaussian_noise"]["delta"]


def test_decision_layer_unavailable_payload_is_explicit() -> None:
    payload = decision_layer_unavailable_payload("missing calibration artifacts")
    assert payload["decision_layer_available"] is False
    assert payload["warning"] == "missing calibration artifacts"
    assert payload["decision_delta_summary"] == []


def test_decision_layer_robustness_returns_expected_metrics() -> None:
    signals = np.random.randn(4, 12, 1000).astype(np.float32)
    labels = np.array(
        [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1],
        ],
        dtype=np.int64,
    )

    def predict_proba_fn(batch: np.ndarray) -> np.ndarray:
        return np.tile(np.array([[0.8, 0.2, 0.2, 0.2, 0.2]], dtype=np.float32), (len(batch), 1))

    results = evaluate_decision_layer_under_perturbations(
        signals,
        labels,
        predict_proba_fn,
        {"gaussian_noise": {"std": 0.01}},
        fn_weights=np.array([1.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float32),
        fp_weights=np.ones(5, dtype=np.float32),
        thresholds={"0.95": 1.0},
        abstain_cost=0.5,
    )
    assert results["decision_layer_available"] is True
    assert len(results["decision_delta_summary"]) == 1
    assert "delta_expected_harm_with_abstain" in results["gaussian_noise"]["delta"]
