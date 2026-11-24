from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ecg_reliability.decision.selective import evaluate_selective_policy
from ecg_reliability.evaluation.calibration_metrics import compute_calibration_report
from ecg_reliability.evaluation.metrics import compute_classification_metrics
from ecg_reliability.robustness.perturbations import apply_perturbation


def _delta_summary(clean: dict[str, Any], perturbed: dict[str, Any]) -> dict[str, float]:
    return {
        "delta_macro_auroc": float(perturbed["classification"]["macro_auroc"] - clean["classification"]["macro_auroc"]),
        "delta_macro_auprc": float(perturbed["classification"]["macro_auprc"] - clean["classification"]["macro_auprc"]),
        "delta_abnormal_macro_auroc": float(
            perturbed["classification"]["abnormal_macro_auroc"] - clean["classification"]["abnormal_macro_auroc"]
        ),
        "delta_abnormal_macro_auprc": float(
            perturbed["classification"]["abnormal_macro_auprc"] - clean["classification"]["abnormal_macro_auprc"]
        ),
        "delta_macro_ece": float(perturbed["calibration"]["macro_ece"] - clean["calibration"]["macro_ece"]),
        "delta_macro_brier": float(perturbed["calibration"]["macro_brier"] - clean["calibration"]["macro_brier"]),
    }


def evaluate_under_perturbations(
    signals: np.ndarray,
    labels: np.ndarray,
    predict_proba_fn: Callable[[np.ndarray], np.ndarray],
    perturbation_specs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    clean_probs = predict_proba_fn(signals)
    clean = {
        "classification": compute_classification_metrics(labels, clean_probs),
        "calibration": compute_calibration_report(labels, clean_probs),
    }
    results = {"clean": clean, "delta_summary": []}
    for name, kwargs in perturbation_specs.items():
        perturbed = np.stack([apply_perturbation(signal, name, **kwargs) for signal in signals], axis=0)
        probs = predict_proba_fn(perturbed)
        perturbed_result = {
            "classification": compute_classification_metrics(labels, probs),
            "calibration": compute_calibration_report(labels, probs),
        }
        delta = _delta_summary(clean, perturbed_result)
        results[name] = {
            **perturbed_result,
            "delta": delta,
        }
        results["delta_summary"].append({"perturbation": name, **delta})
    return results


def decision_layer_unavailable_payload(reason: str) -> dict[str, Any]:
    return {
        "decision_layer_available": False,
        "warning": reason,
        "clean": None,
        "decision_delta_summary": [],
    }


def _select_curve_summary(curves: list[dict[str, Any]], preferred_coverage: float = 0.95) -> dict[str, float]:
    if not curves:
        raise ValueError("Selective policy results must contain at least one curve.")
    selected = min(
        curves,
        key=lambda row: (abs(float(row["target_coverage"]) - preferred_coverage), float(row["target_coverage"])),
    )
    return {
        "target_coverage": float(selected["target_coverage"]),
        "coverage": float(selected["coverage"]),
        "accepted_risk": float(selected["accepted_risk"]),
        "expected_harm_with_abstain": float(selected["expected_harm_with_abstain"]),
        "abstain_rate": float(selected["abstain_rate"]),
        "weighted_fn_harm": float(selected["weighted_fn_harm"]),
        "weighted_fp_harm": float(selected["weighted_fp_harm"]),
    }


def _decision_delta_summary(clean: dict[str, float], perturbed: dict[str, float]) -> dict[str, float]:
    return {
        "delta_expected_harm_with_abstain": float(perturbed["expected_harm_with_abstain"] - clean["expected_harm_with_abstain"]),
        "delta_abstain_rate": float(perturbed["abstain_rate"] - clean["abstain_rate"]),
        "delta_accepted_risk": float(perturbed["accepted_risk"] - clean["accepted_risk"]),
        "delta_weighted_fn_harm": float(perturbed["weighted_fn_harm"] - clean["weighted_fn_harm"]),
        "delta_weighted_fp_harm": float(perturbed["weighted_fp_harm"] - clean["weighted_fp_harm"]),
    }


def evaluate_decision_layer_under_perturbations(
    signals: np.ndarray,
    labels: np.ndarray,
    predict_probabilities_fn: Callable[[np.ndarray], np.ndarray],
    perturbation_specs: dict[str, dict[str, Any]],
    *,
    fn_weights: np.ndarray,
    fp_weights: np.ndarray,
    thresholds: dict[str, float],
    abstain_cost: float,
    preferred_coverage: float = 0.95,
) -> dict[str, Any]:
    clean_probs = predict_probabilities_fn(signals)
    clean_policy = evaluate_selective_policy(
        clean_probs,
        labels,
        fn_weights=fn_weights,
        fp_weights=fp_weights,
        thresholds=thresholds,
        abstain_cost=abstain_cost,
    )
    clean_summary = _select_curve_summary(clean_policy["curves"], preferred_coverage=preferred_coverage)
    results: dict[str, Any] = {
        "decision_layer_available": True,
        "preferred_coverage": float(preferred_coverage),
        "target_coverage_used": float(clean_summary["target_coverage"]),
        "clean": clean_summary,
        "decision_delta_summary": [],
    }
    for name, kwargs in perturbation_specs.items():
        perturbed = np.stack([apply_perturbation(signal, name, **kwargs) for signal in signals], axis=0)
        probs = predict_probabilities_fn(perturbed)
        policy = evaluate_selective_policy(
            probs,
            labels,
            fn_weights=fn_weights,
            fp_weights=fp_weights,
            thresholds=thresholds,
            abstain_cost=abstain_cost,
        )
        perturbed_summary = _select_curve_summary(policy["curves"], preferred_coverage=preferred_coverage)
        delta = _decision_delta_summary(clean_summary, perturbed_summary)
        results[name] = {
            **perturbed_summary,
            "delta": delta,
        }
        results["decision_delta_summary"].append(
            {
                "perturbation": name,
                "target_coverage": float(perturbed_summary["target_coverage"]),
                **delta,
            }
        )
    return results
