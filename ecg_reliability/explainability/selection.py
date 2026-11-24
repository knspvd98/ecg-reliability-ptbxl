from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ecg_reliability.data.labels import ABNORMAL_LABELS, TARGET_LABELS
from ecg_reliability.decision.consistency import apply_norm_consistency_rule
from ecg_reliability.decision.selective import uncertainty_scores


def resolve_preferred_coverage_key(thresholds: dict[str, float], preferred: float = 0.95) -> str:
    if not thresholds:
        raise ValueError("Thresholds are required to resolve a preferred coverage key.")
    return min(thresholds.keys(), key=lambda key: (abs(float(key) - preferred), float(key)))


def _accepted_label_names(decision_row: np.ndarray) -> list[str]:
    return [label for label, value in zip(TARGET_LABELS, decision_row, strict=True) if int(value) == 1]


def build_xai_selection_manifests(
    ecg_ids: np.ndarray,
    probabilities: np.ndarray,
    *,
    num_samples: int,
    thresholds: dict[str, float] | None = None,
    fn_weights: np.ndarray | None = None,
    fp_weights: np.ndarray | None = None,
    preferred_coverage: float = 0.95,
    raw_argmax_debug_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    sample_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    selected_sample_ids: set[int] = set()

    if thresholds is not None and fn_weights is not None and fp_weights is not None:
        raw_decisions, uncertainty = uncertainty_scores(probabilities, fn_weights, fp_weights)
        decisions = apply_norm_consistency_rule(raw_decisions)
        coverage_key = resolve_preferred_coverage_key(thresholds, preferred=preferred_coverage)
        threshold = float(thresholds[coverage_key])
        accept_mask = uncertainty <= threshold
        decision_source = "calibrated_decision"

        candidate_rows: list[dict[str, Any]] = []
        for idx, ecg_id in enumerate(ecg_ids):
            accepted_labels = _accepted_label_names(decisions[idx])
            abnormal_targets = [label for label in accepted_labels if label in ABNORMAL_LABELS]
            abstained = bool(not accept_mask[idx])
            if abstained:
                case_type = "abstained"
                targets_for_sample: list[str] = []
            elif abnormal_targets:
                case_type = "abnormal"
                targets_for_sample = abnormal_targets
            elif "NORM" in accepted_labels:
                case_type = "norm_only"
                targets_for_sample = ["NORM"]
            else:
                case_type = "empty_decision"
                targets_for_sample = []
            row = {
                "ecg_id": int(ecg_id),
                "decision_source": decision_source,
                "abstained": abstained,
                "accepted_labels": "|".join(accepted_labels),
                "case_type": case_type,
                "selected_for_xai": False,
                "target_labels": "|".join(targets_for_sample),
            }
            sample_rows.append(row)
            if targets_for_sample:
                candidate_rows.append(row)

        for row in candidate_rows[: max(int(num_samples), 0)]:
            selected_sample_ids.add(int(row["ecg_id"]))

        for row in sample_rows:
            row["selected_for_xai"] = int(row["ecg_id"]) in selected_sample_ids
            if row["selected_for_xai"]:
                targets_for_sample = [label for label in row["target_labels"].split("|") if label]
                for target_label in targets_for_sample:
                    target_rows.append(
                        {
                            "ecg_id": int(row["ecg_id"]),
                            "target_label": target_label,
                            "case_type": row["case_type"],
                            "decision_source": decision_source,
                        }
                    )
        summary = {
            "decision_source": decision_source,
            "preferred_coverage": float(preferred_coverage),
            "coverage_key_used": coverage_key,
            "threshold_used": threshold,
            "warnings": warnings,
            "abstained_count": int((~accept_mask).sum()),
            "selected_sample_count": int(len(selected_sample_ids)),
            "selected_target_count": int(len(target_rows)),
        }
    else:
        warnings.append("raw_argmax_debug_only")
        argmax_indices = np.argmax(probabilities, axis=1)
        for idx, ecg_id in enumerate(ecg_ids):
            target_label = TARGET_LABELS[int(argmax_indices[idx])]
            case_type = "norm_only" if target_label == "NORM" else "abnormal"
            sample_rows.append(
                {
                    "ecg_id": int(ecg_id),
                    "decision_source": "raw_argmax_debug_only",
                    "abstained": False,
                    "accepted_labels": target_label,
                    "case_type": case_type,
                    "selected_for_xai": idx < max(int(num_samples), 0),
                    "target_labels": target_label,
                }
            )
            if idx < max(int(num_samples), 0):
                target_rows.append(
                    {
                        "ecg_id": int(ecg_id),
                        "target_label": target_label,
                        "case_type": case_type,
                        "decision_source": "raw_argmax_debug_only",
                    }
                )
        summary = {
            "decision_source": "raw_argmax_debug_only",
            "preferred_coverage": None,
            "coverage_key_used": None,
            "threshold_used": None,
            "warnings": warnings,
            "abstained_count": 0,
            "selected_sample_count": int(min(len(ecg_ids), max(int(num_samples), 0))),
            "selected_target_count": int(len(target_rows)),
        }

    return pd.DataFrame(sample_rows), pd.DataFrame(target_rows), summary
