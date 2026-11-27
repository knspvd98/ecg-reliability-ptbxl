from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from ecg_reliability.evaluation.bootstrap import bootstrap_metric, paired_bootstrap_comparison
from ecg_reliability.evaluation.calibration_metrics import classwise_ece
from ecg_reliability.evaluation.failure_analysis import analyze_model_disagreement, analyze_model_failures
from ecg_reliability.explainability.integrated_gradients import compute_integrated_gradients
from ecg_reliability.explainability.sanity_checks import (
    faithfulness_deletion_insertion,
    ig_completeness_check,
    model_parameter_randomization_check,
)
from ecg_reliability.models.simple_cnn import SimpleCNN
from ecg_reliability.robustness.perturbations import (
    apply_perturbation,
    build_severity_perturbation_suite,
    electrode_motion,
    muscle_noise,
    powerline_interference,
)


def test_patient_cluster_bootstrap_and_invalid_resample_tracking() -> None:
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    y_score = np.array([0.1, 0.2, 0.15, 0.3, 0.8, 0.9, 0.85, 0.75], dtype=np.float32)
    clusters = np.array([101, 101, 102, 102, 201, 201, 202, 202], dtype=np.int64)

    res = bootstrap_metric(
        y_true,
        y_score,
        lambda yt, ys: float(roc_auc_score(yt, ys)),
        n_bootstrap=60,
        seed=42,
        cluster_ids=clusters,
        ci_level=0.95,
    )
    assert res["resampling_unit"] == "patient_cluster"
    assert res["n_records"] == 8
    assert res["n_clusters"] == 4
    assert res["effective_sample_size"] == 4
    assert res["ci_definition"] == "percentile_95pct_patient_cluster"
    assert res["n_bootstrap_valid"] + res["n_bootstrap_invalid"] == 60
    assert res["valid_ci"] is True

    # Degenerate all-zero labels -> every AUROC resample is invalid and explicitly reported
    degenerate = bootstrap_metric(
        np.zeros(6, dtype=np.int64),
        np.linspace(0.1, 0.9, 6),
        lambda yt, ys: float(roc_auc_score(yt, ys)),
        n_bootstrap=15,
        seed=0,
    )
    assert degenerate["valid_ci"] is False
    assert degenerate["n_bootstrap_valid"] == 0
    assert degenerate["n_bootstrap_invalid"] == 15
    assert degenerate["invalid_resample_rate"] == 1.0


def test_paired_bootstrap_comparison_detects_superior_model() -> None:
    y_true = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=np.int64)
    clusters = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6], dtype=np.int64)
    good_scores = np.array([0.05, 0.10, 0.12, 0.15, 0.20, 0.25, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95])
    weak_scores = np.array([0.45, 0.55, 0.60, 0.40, 0.52, 0.48, 0.52, 0.49, 0.55, 0.45, 0.58, 0.51])

    comp = paired_bootstrap_comparison(
        y_true,
        good_scores,
        weak_scores,
        lambda yt, ys: float(roc_auc_score(yt, ys)),
        n_bootstrap=80,
        seed=42,
        cluster_ids=clusters,
    )
    assert comp["valid_ci"] is True
    assert comp["delta_point_estimate"] > 0.2
    assert comp["delta_ci_low"] > 0.0
    assert comp["resampling_unit"] == "patient_cluster"


def test_calibration_rejects_degenerate_and_out_of_range_inputs() -> None:
    with pytest.raises(ValueError, match="finite probabilities"):
        classwise_ece(np.array([[0, 1]]), np.array([[1.2, 0.5]]))
    with pytest.raises(ValueError, match="matching, non-empty"):
        classwise_ece(np.empty((0, 5)), np.empty((0, 5)))
    with pytest.raises(ValueError, match="binary targets"):
        classwise_ece(np.array([[0, 2]]), np.array([[0.2, 0.8]]))


def test_realistic_ecg_perturbations_and_severity_suite() -> None:
    t = np.arange(1000, dtype=np.float32) / 100.0
    clean = np.tile(np.sin(2.0 * np.pi * 1.2 * t), (12, 1)).astype(np.float32)

    pl = powerline_interference(clean, amplitude=0.2, frequency_hz=50.0, sampling_rate=100)
    emg = muscle_noise(clean, snr_db=6.0, sampling_rate=100, seed=42)
    em = electrode_motion(clean, amplitude=0.8, n_events=2, sampling_rate=100, seed=42)
    for arr in (pl, emg, em):
        assert arr.shape == clean.shape
        assert np.isfinite(arr).all()
        assert not np.allclose(arr, clean)

    suite = build_severity_perturbation_suite(sampling_rate=100, seed=42)
    assert len(suite) >= 10
    for name, spec in suite.items():
        out = apply_perturbation(clean, name, **spec)
        assert out.shape == clean.shape
        assert np.isfinite(out).all()


def test_attribution_sanity_checks_and_failure_disagreement_analysis() -> None:
    torch.manual_seed(0)
    model = SimpleCNN(in_channels=12, num_classes=5, dropout=0.0).eval()
    x = torch.randn(1, 12, 1000)
    ig = compute_integrated_gradients(model, x, target_class=1, steps=16)

    comp = ig_completeness_check(model, x, ig, target_class=1)
    assert np.isfinite(comp["abs_error"])
    assert comp["abs_error"] < 0.25

    rand_res = model_parameter_randomization_check(
        model,
        x,
        target_class=1,
        attr_fn=lambda m, inp, cls: compute_integrated_gradients(m, inp, target_class=cls, steps=8),
        seed=7,
    )
    assert np.isfinite(rand_res["full_randomized_spearman"])
    assert np.isfinite(rand_res["head_randomized_cosine"])

    faith = faithfulness_deletion_insertion(model, x, ig.numpy()[0], target_class=1, steps=5)
    assert 0.0 <= faith["deletion_auc"] <= 1.0
    assert 0.0 <= faith["insertion_auc"] <= 1.0

    ids = np.array([10, 11, 12], dtype=np.int64)
    y_true = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]], dtype=np.int64)
    probs_a = np.array([[0.1, 0.9, 0.1, 0.1, 0.1], [0.1, 0.85, 0.1, 0.1, 0.1], [0.2, 0.2, 0.8, 0.1, 0.1]])
    probs_b = np.array([[0.9, 0.1, 0.1, 0.1, 0.1], [0.1, 0.80, 0.1, 0.1, 0.1], [0.7, 0.1, 0.2, 0.1, 0.1]])
    failures = analyze_model_failures(ids, y_true, probs_a)
    assert len(failures["high_confidence_false_positives"]) >= 1
    assert len(failures["high_confidence_false_negatives"]) >= 1

    dis = analyze_model_disagreement(
        {
            "model_a": {"ecg_ids": ids, "probabilities": probs_a, "labels": y_true},
            "model_b": {"ecg_ids": ids, "probabilities": probs_b, "labels": y_true},
        }
    )
    assert dis["shared_ecg_count"] == 3
    assert len(dis["top_disagreement_cases"]) == 3
    assert dis["top_disagreement_cases"][0]["ecg_id"] == 10
