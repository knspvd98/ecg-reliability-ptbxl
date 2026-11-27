from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np

from ecg_reliability.data.labels import TARGET_LABELS
from ecg_reliability.decision.consistency import apply_norm_consistency_rule
from ecg_reliability.decision.harm import realized_harm
from ecg_reliability.decision.selective import uncertainty_scores


def analyze_model_failures(
    ecg_ids: np.ndarray,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    label_names: list[str] | None = None,
    decision_threshold: float = 0.5,
    fn_weights: np.ndarray | None = None,
    fp_weights: np.ndarray | None = None,
    selective_threshold: float | None = None,
    top_k: int = 20,
) -> dict[str, Any]:
    """Identify misclassified cases, high-confidence errors, and selective-prediction failures."""
    labels = list(label_names or TARGET_LABELS)
    ids = np.asarray(ecg_ids, dtype=np.int64)
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)

    if fn_weights is not None and fp_weights is not None:
        raw_dec, uncertainty = uncertainty_scores(p, fn_weights, fp_weights)
        preds = apply_norm_consistency_rule(raw_dec, label_names=labels)
        sample_harm = realized_harm(preds, y, fn_weights, fp_weights)
        abstained = (uncertainty > float(selective_threshold)) if selective_threshold is not None else np.zeros(len(ids), dtype=bool)
    else:
        preds = (p >= float(decision_threshold)).astype(int)
        uncertainty = np.mean(1.0 - np.abs(2.0 * p - 1.0), axis=1)
        sample_harm = np.sum(preds != y, axis=1).astype(np.float64)
        abstained = np.zeros(len(ids), dtype=bool)

    high_conf_fp: list[dict[str, Any]] = []
    high_conf_fn: list[dict[str, Any]] = []
    per_class_summary: dict[str, dict[str, int | float]] = {}

    for cls_idx, label in enumerate(labels):
        yt = y[:, cls_idx]
        yp = preds[:, cls_idx]
        pr = p[:, cls_idx]
        tp = int(((yt == 1) & (yp == 1)).sum())
        tn = int(((yt == 0) & (yp == 0)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        per_class_summary[label] = {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "support_positive": int((yt == 1).sum()),
            "mean_prob_positive": float(pr[yt == 1].mean()) if np.any(yt == 1) else float("nan"),
            "mean_prob_negative": float(pr[yt == 0].mean()) if np.any(yt == 0) else float("nan"),
        }
        for i in np.flatnonzero((yt == 0) & (yp == 1)):
            high_conf_fp.append(
                {
                    "ecg_id": int(ids[i]),
                    "label": label,
                    "error_type": "false_positive",
                    "predicted_prob": float(pr[i]),
                    "true_labels": [labels[j] for j in range(len(labels)) if y[i, j] == 1],
                    "predicted_labels": [labels[j] for j in range(len(labels)) if preds[i, j] == 1],
                    "uncertainty": float(uncertainty[i]),
                    "realized_harm": float(sample_harm[i]),
                    "abstained": bool(abstained[i]),
                }
            )
        for i in np.flatnonzero((yt == 1) & (yp == 0)):
            high_conf_fn.append(
                {
                    "ecg_id": int(ids[i]),
                    "label": label,
                    "error_type": "false_negative",
                    "predicted_prob": float(pr[i]),
                    "confidence_in_wrong_neg": float(1.0 - pr[i]),
                    "true_labels": [labels[j] for j in range(len(labels)) if y[i, j] == 1],
                    "predicted_labels": [labels[j] for j in range(len(labels)) if preds[i, j] == 1],
                    "uncertainty": float(uncertainty[i]),
                    "realized_harm": float(sample_harm[i]),
                    "abstained": bool(abstained[i]),
                }
            )

    high_conf_fp.sort(key=lambda r: -r["predicted_prob"])
    high_conf_fn.sort(key=lambda r: r["predicted_prob"])

    record_errors = np.any(preds != y, axis=1)
    abstained_cases = [
        {
            "ecg_id": int(ids[i]),
            "true_labels": [labels[j] for j in range(len(labels)) if y[i, j] == 1],
            "predicted_labels": [labels[j] for j in range(len(labels)) if preds[i, j] == 1],
            "probabilities": {labels[j]: float(p[i, j]) for j in range(len(labels))},
            "uncertainty": float(uncertainty[i]),
            "would_have_errored": bool(record_errors[i]),
        }
        for i in np.flatnonzero(abstained)
    ]
    abstained_cases.sort(key=lambda r: -r["uncertainty"])

    return {
        "n_records": int(len(ids)),
        "n_records_with_any_error": int(record_errors.sum()),
        "exact_match_accuracy": float((~record_errors).mean()) if len(ids) else float("nan"),
        "n_abstained": int(abstained.sum()),
        "per_class_confusion": per_class_summary,
        "high_confidence_false_positives": high_conf_fp[: int(top_k)],
        "high_confidence_false_negatives": high_conf_fn[: int(top_k)],
        "abstained_uncertain_cases": abstained_cases[: int(top_k)],
    }


def analyze_model_disagreement(
    model_predictions: dict[str, dict[str, np.ndarray]],
    *,
    label_names: list[str] | None = None,
    threshold: float = 0.5,
    top_k: int = 20,
) -> dict[str, Any]:
    """Compare multiple models on shared ECG records to surface inter-model disagreement."""
    labels = list(label_names or TARGET_LABELS)
    model_names = sorted(model_predictions.keys())
    if len(model_names) < 2:
        return {
            "models": model_names,
            "shared_ecg_count": 0,
            "pairwise_summary": [],
            "top_disagreement_cases": [],
        }

    # Intersect ecg_ids across all models
    id_sets = [set(np.asarray(model_predictions[m]["ecg_ids"], dtype=np.int64).tolist()) for m in model_names]
    shared_ids = sorted(set.intersection(*id_sets))
    if not shared_ids:
        return {
            "models": model_names,
            "shared_ecg_count": 0,
            "pairwise_summary": [],
            "top_disagreement_cases": [],
        }

    aligned_probs: dict[str, np.ndarray] = {}
    y_true_shared: np.ndarray | None = None
    for name in model_names:
        ids = np.asarray(model_predictions[name]["ecg_ids"], dtype=np.int64)
        idx_map = {int(eid): pos for pos, eid in enumerate(ids)}
        order = [idx_map[eid] for eid in shared_ids]
        aligned_probs[name] = np.asarray(model_predictions[name]["probabilities"], dtype=np.float64)[order]
        if y_true_shared is None and "labels" in model_predictions[name]:
            y_true_shared = np.asarray(model_predictions[name]["labels"], dtype=np.int64)[order]

    pairwise_summary: list[dict[str, Any]] = []
    for m_a, m_b in combinations(model_names, 2):
        pa = aligned_probs[m_a]
        pb = aligned_probs[m_b]
        da = (pa >= threshold).astype(int)
        db = (pb >= threshold).astype(int)
        pairwise_summary.append(
            {
                "model_a": m_a,
                "model_b": m_b,
                "mean_abs_prob_diff": float(np.mean(np.abs(pa - pb))),
                "max_abs_prob_diff": float(np.max(np.abs(pa - pb))),
                "decision_agreement_rate": float(np.mean(da == db)),
                "exact_record_agreement_rate": float(np.mean(np.all(da == db, axis=1))),
            }
        )

    stacked = np.stack([aligned_probs[m] for m in model_names], axis=0)  # (M, N, K)
    prob_std = np.std(stacked, axis=0)  # (N, K)
    record_disagreement = np.max(prob_std, axis=1)  # (N,)
    order = np.argsort(-record_disagreement)

    top_cases: list[dict[str, Any]] = []
    for idx in order[: int(top_k)]:
        disputed_cls = int(np.argmax(prob_std[idx]))
        per_model = {
            m: {labels[k]: float(aligned_probs[m][idx, k]) for k in range(len(labels))}
            for m in model_names
        }
        top_cases.append(
            {
                "ecg_id": int(shared_ids[idx]),
                "max_prob_std": float(record_disagreement[idx]),
                "most_disputed_label": labels[disputed_cls],
                "true_labels": (
                    [labels[k] for k in range(len(labels)) if y_true_shared[idx, k] == 1]
                    if y_true_shared is not None
                    else []
                ),
                "model_probabilities": per_model,
            }
        )

    return {
        "models": model_names,
        "shared_ecg_count": int(len(shared_ids)),
        "pairwise_summary": pairwise_summary,
        "top_disagreement_cases": top_cases,
    }
