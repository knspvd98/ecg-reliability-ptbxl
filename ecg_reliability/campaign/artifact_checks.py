from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ecg_reliability.evaluation.model_selection import load_selection_manifest

INTERNAL_CORE_ARTIFACTS = [
    "outputs/index/index_summary.json",
    "outputs/index/label_cooccurrence.csv",
    "outputs/cache/cache_summary.json",
    "outputs/summary/runtime_preflight.json",
    "outputs/summary/model_selection.json",
    "outputs/summary/results_summary.json",
    "outputs/campaign/manifest.json",
    "outputs/campaign/stage_state.json",
]


def _resolve_artifact(path: str | Path, outputs_root: str | Path) -> Path:
    candidate = Path(path)
    if str(candidate).startswith("outputs/"):
        return Path(outputs_root) / Path(*candidate.parts[1:])
    return candidate


def _artifact_exists(path: str | Path, outputs_root: str | Path) -> bool:
    return _resolve_artifact(path, outputs_root).exists()


def completed_internal_models(outputs_root: str | Path = "outputs") -> list[str]:
    root = Path(outputs_root) / "predictions"
    models: list[str] = []
    if not root.exists():
        return models
    for metrics_path in root.glob("*/raw/test_metrics_raw.json"):
        model_name = metrics_path.parts[-3]
        if model_name not in models:
            models.append(model_name)
    return sorted(models)


def _model_artifacts(model_name: str) -> list[str]:
    checkpoint = "model.pkl" if model_name == "xgboost" else "best.pt"
    return [
        f"outputs/checkpoints/{model_name}/{checkpoint}",
        f"outputs/predictions/{model_name}/raw/test_metrics_raw.json",
        f"outputs/calibration/{model_name}/reports.json",
    ]


def _decision_artifact(model_name: str, outputs_root: str | Path) -> str | None:
    decision_root = Path(outputs_root) / "decision" / model_name
    if not decision_root.exists():
        return None
    for candidate in sorted(decision_root.glob("*/results.json")):
        return str(candidate)
    return None


def _selected_model_artifacts(selected_model: str) -> list[str]:
    return [
        f"outputs/robustness/{selected_model}/results.json",
        f"outputs/explanations/{selected_model}/stability.csv",
        "outputs/figures/figure_1_pipeline.png",
        "outputs/figures/figure_2_dataset_labels.png",
        "outputs/figures/figure_3_model_comparison.png",
        "outputs/figures/figure_4_calibration.png",
        "outputs/figures/figure_4_expected_harm.png",
        "outputs/figures/figure_4_risk_coverage.png",
        "outputs/figures/figure_5_xai.png",
        "outputs/tables/table_1_label_distribution.csv",
        "outputs/tables/table_2_model_raw_metrics.csv",
        "outputs/tables/table_3_calibration_comparison.csv",
        "outputs/tables/table_4_selective_policy.csv",
        "outputs/tables/table_5_robustness_deltas.csv",
        "outputs/tables/table_6_xai_stability.csv",
    ]


def validate_claim_artifacts(
    *,
    selected_model: str,
    require_external: bool,
    external_enabled: bool,
    outputs_root: str | Path = "outputs",
) -> dict[str, Any]:
    outputs_root = str(outputs_root)
    missing_internal: list[str] = []
    checked_internal: list[str] = []
    for artifact in INTERNAL_CORE_ARTIFACTS:
        checked_internal.append(artifact)
        if not _artifact_exists(artifact, outputs_root):
            missing_internal.append(artifact)

    models = completed_internal_models(outputs_root)
    for model_name in models:
        for artifact in _model_artifacts(model_name):
            checked_internal.append(artifact)
            if not _artifact_exists(artifact, outputs_root):
                missing_internal.append(artifact)
        decision_artifact = _decision_artifact(model_name, outputs_root)
        if decision_artifact is None:
            missing_internal.append(f"outputs/decision/{model_name}/*/results.json")
        else:
            checked_internal.append(decision_artifact)

    for artifact in _selected_model_artifacts(selected_model):
        checked_internal.append(artifact)
        if not _artifact_exists(artifact, outputs_root):
            missing_internal.append(artifact)

    require_external = bool(require_external or external_enabled or os.environ.get("INCLUDE_EXTERNAL") == "1")
    external_artifacts = [
        "outputs/external/external_manifest.parquet",
        "outputs/external/external_cache_summary.json",
        f"outputs/external/{selected_model}/external_binary_results.json",
        f"outputs/external/{selected_model}/external_sourcewise_results.csv",
    ]
    missing_external: list[str] = []
    if require_external:
        for artifact in external_artifacts:
            if not _artifact_exists(artifact, outputs_root):
                missing_external.append(artifact)
        external_table = "outputs/tables/table_7_external_binary_validation.csv"
        if not _artifact_exists(external_table, outputs_root):
            missing_external.append(external_table)

    if os.environ.get("RUN_BOOTSTRAP") == "1" and not _artifact_exists("outputs/tables/table_bootstrap_classification_ci.csv", outputs_root):
        missing_internal.append("outputs/tables/table_bootstrap_classification_ci.csv")

    overlap_path = Path(f"outputs/external/{selected_model}/snomed_overlap_results.json")
    payload = {
        "selected_deep_model": selected_model,
        "completed_internal_models": models,
        "checked_internal_artifacts": checked_internal,
        "missing_internal_artifacts": missing_internal,
        "missing_external_artifacts": missing_external,
        "internal_ptbxl_validation_passed": not missing_internal,
        "external_binary_validation_passed": bool(require_external and not missing_external),
        "external_snomed_overlap_validation_passed": bool(overlap_path.exists()),
        "external_five_superclass_validation_passed": False,
        "external_required": require_external,
    }
    return payload


def selected_model_from_manifest(path: str | Path) -> str:
    payload = load_selection_manifest(path)
    return str(payload["selected_deep_model"])
