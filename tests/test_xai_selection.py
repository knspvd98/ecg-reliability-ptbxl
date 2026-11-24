from __future__ import annotations

import numpy as np

from ecg_reliability.explainability.selection import build_xai_selection_manifests


def test_xai_target_selection_uses_accepted_labels_and_excludes_abstained() -> None:
    ecg_ids = np.array([101, 102, 103], dtype=np.int64)
    probabilities = np.array(
        [
            [0.1, 0.9, 0.2, 0.1, 0.1],
            [0.9, 0.1, 0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2, 0.2, 0.2],
        ],
        dtype=np.float32,
    )
    sample_manifest, target_manifest, summary = build_xai_selection_manifests(
        ecg_ids=ecg_ids,
        probabilities=probabilities,
        num_samples=3,
        thresholds={"0.95": 1.0, "0.99": 10.0},
        fn_weights=np.array([1.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float32),
        fp_weights=np.ones(5, dtype=np.float32),
    )
    assert summary["decision_source"] == "calibrated_decision"
    assert "abstained" in sample_manifest["case_type"].tolist()
    selected_targets = target_manifest.loc[target_manifest["ecg_id"] == 101, "target_label"].tolist()
    assert selected_targets == ["MI"]
    assert 103 not in target_manifest["ecg_id"].tolist()


def test_raw_argmax_fallback_is_marked_debug_only() -> None:
    ecg_ids = np.array([201, 202], dtype=np.int64)
    probabilities = np.array(
        [
            [0.6, 0.2, 0.1, 0.05, 0.05],
            [0.1, 0.7, 0.1, 0.05, 0.05],
        ],
        dtype=np.float32,
    )
    sample_manifest, target_manifest, summary = build_xai_selection_manifests(
        ecg_ids=ecg_ids,
        probabilities=probabilities,
        num_samples=2,
    )
    assert summary["decision_source"] == "raw_argmax_debug_only"
    assert "raw_argmax_debug_only" in summary["warnings"]
    assert sample_manifest["decision_source"].nunique() == 1
    assert target_manifest["decision_source"].nunique() == 1
