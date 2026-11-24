from __future__ import annotations

import numpy as np

from ecg_reliability.decision.consistency import apply_norm_consistency_rule, norm_conflict_report
from ecg_reliability.decision.selective import (
    build_selective_policy_checks,
    evaluate_selective_policy,
    fit_coverage_thresholds,
)


def test_norm_consistency_rule_suppresses_normal_when_abnormal_present() -> None:
    predictions = np.array([[1, 1, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 1, 0]])
    adjusted = apply_norm_consistency_rule(predictions)
    assert adjusted[0, 0] == 0
    assert adjusted[1, 0] == 1
    assert adjusted[2, 0] == 0


def test_selective_policy_has_monotonic_coverage() -> None:
    probabilities = np.array(
        [
            [0.95, 0.10, 0.10, 0.10, 0.10],
            [0.15, 0.80, 0.20, 0.10, 0.10],
            [0.20, 0.25, 0.75, 0.15, 0.10],
            [0.10, 0.10, 0.10, 0.85, 0.80],
            [0.45, 0.40, 0.30, 0.20, 0.20],
        ],
        dtype=np.float32,
    )
    labels = np.array(
        [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1],
            [1, 0, 0, 0, 0],
        ],
        dtype=np.int64,
    )
    fn_weights = np.array([1.0, 2.5, 2.0, 2.0, 2.0], dtype=np.float32)
    fp_weights = np.ones(5, dtype=np.float32)
    thresholds = fit_coverage_thresholds(probabilities, labels, fn_weights, fp_weights, [0.6, 0.8, 1.0])
    results = evaluate_selective_policy(probabilities, labels, fn_weights, fp_weights, thresholds["thresholds"], abstain_cost=0.5)
    coverages = [item["coverage"] for item in results["curves"]]
    assert results["coverage_monotonic"] is True
    assert coverages == sorted(coverages)
    assert "checks" in results
    assert results["checks"]["max_abs_coverage_gap"] >= 0.0


def test_norm_conflict_report_counts_are_correct() -> None:
    raw = np.array([[1, 1, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 1, 0]])
    adjusted = apply_norm_consistency_rule(raw)
    report = norm_conflict_report(raw, adjusted)
    assert report["raw_conflict_count"] == 2
    assert report["post_conflict_count"] == 0
    assert report["n_changed_norm_decisions"] == 2


def test_selective_policy_checks_handle_unsorted_targets() -> None:
    curves = [
        {"target_coverage": 0.99, "coverage": 1.0},
        {"target_coverage": 0.9, "coverage": 0.8},
        {"target_coverage": 0.95, "coverage": 0.9},
    ]
    checks = build_selective_policy_checks(curves)
    assert checks["coverage_monotonic"] is True
    assert checks["target_coverage_values"] == [0.9, 0.95, 0.99]
    assert checks["achieved_coverage_values"] == [0.8, 0.9, 1.0]
