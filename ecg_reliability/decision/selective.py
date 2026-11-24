from __future__ import annotations

from typing import Any

import numpy as np

from ecg_reliability.decision.consistency import apply_norm_consistency_rule, norm_conflict_report
from ecg_reliability.decision.harm import minimal_expected_harm, realized_harm


def uncertainty_scores(probabilities: np.ndarray, fn_weights: np.ndarray, fp_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    decisions, min_harm = minimal_expected_harm(probabilities, fn_weights, fp_weights)
    total_min_harm = min_harm.sum(axis=1)
    return decisions, total_min_harm


def fit_coverage_thresholds(
    probabilities: np.ndarray,
    labels: np.ndarray,
    fn_weights: np.ndarray,
    fp_weights: np.ndarray,
    coverage_targets: list[float],
) -> dict[str, Any]:
    decisions, uncertainty = uncertainty_scores(probabilities, fn_weights, fp_weights)
    decisions = apply_norm_consistency_rule(decisions)
    order = np.argsort(uncertainty)
    sorted_uncertainty = uncertainty[order]
    thresholds = {}
    target_values = sorted(float(coverage) for coverage in coverage_targets)
    achieved_coverages = []
    for coverage in target_values:
        accepted = max(1, int(np.floor(len(sorted_uncertainty) * coverage)))
        threshold = float(sorted_uncertainty[accepted - 1])
        thresholds[str(coverage)] = threshold
        achieved_coverages.append(float((uncertainty <= threshold).mean()))
    return {
        "thresholds": thresholds,
        "sorted_uncertainty": sorted_uncertainty.tolist(),
        "target_coverage_values": target_values,
        "validation_achieved_coverages": achieved_coverages,
    }


def build_selective_policy_checks(curves: list[dict[str, Any]]) -> dict[str, Any]:
    sorted_curves = sorted(curves, key=lambda item: float(item["target_coverage"]))
    target_values = [float(item["target_coverage"]) for item in sorted_curves]
    achieved_values = [float(item["coverage"]) for item in sorted_curves]
    gaps = [
        abs(target - achieved)
        for target, achieved in zip(target_values, achieved_values, strict=True)
    ]
    coverage_monotonic = all(
        achieved_values[idx] <= achieved_values[idx + 1]
        for idx in range(len(achieved_values) - 1)
    )
    max_gap = float(max(gaps)) if gaps else 0.0
    return {
        "coverage_monotonic": coverage_monotonic,
        "target_coverage_values": target_values,
        "achieved_coverage_values": achieved_values,
        "max_abs_coverage_gap": max_gap,
        "warning_large_coverage_gap": bool(max_gap > 0.05),
    }


def evaluate_selective_policy(
    probabilities: np.ndarray,
    labels: np.ndarray,
    fn_weights: np.ndarray,
    fp_weights: np.ndarray,
    thresholds: dict[str, float],
    abstain_cost: float,
) -> dict[str, Any]:
    raw_decisions, uncertainty = uncertainty_scores(probabilities, fn_weights, fp_weights)
    base_decisions = apply_norm_consistency_rule(raw_decisions)
    conflict = norm_conflict_report(raw_decisions, base_decisions)
    realized = realized_harm(base_decisions, labels, fn_weights, fp_weights)
    results = []
    for coverage_key, threshold in sorted(thresholds.items(), key=lambda item: float(item[0])):
        accept_mask = uncertainty <= threshold
        coverage = float(accept_mask.mean())
        accepted_risk = float(realized[accept_mask].mean()) if np.any(accept_mask) else float("nan")
        overall_expected_harm = float(
            np.where(accept_mask, realized, abstain_cost).mean()
        )
        results.append(
            {
                "target_coverage": float(coverage_key),
                "coverage": coverage,
                "coverage_gap": float(abs(float(coverage_key) - coverage)),
                "warning_large_coverage_gap": bool(abs(float(coverage_key) - coverage) > 0.05),
                "accepted_risk": accepted_risk,
                "expected_harm_with_abstain": overall_expected_harm,
                "abstain_rate": float(1.0 - coverage),
                "weighted_fn_harm": float(
                    (((labels == 1) & (base_decisions == 0)).astype(np.float32) * fn_weights)[accept_mask].sum() / max(int(accept_mask.sum()), 1)
                ),
                "weighted_fp_harm": float(
                    (((labels == 0) & (base_decisions == 1)).astype(np.float32) * fp_weights)[accept_mask].sum() / max(int(accept_mask.sum()), 1)
                ),
            }
        )
    checks = build_selective_policy_checks(results)
    return {
        "curves": results,
        "coverage_monotonic": checks["coverage_monotonic"],
        "checks": checks,
        "norm_conflict_report": conflict,
    }
