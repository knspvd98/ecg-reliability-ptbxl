from __future__ import annotations

from pathlib import Path

from ecg_reliability.campaign.artifact_checks import validate_claim_artifacts


def _touch(path: Path, text: str = "{}") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_artifact_validator_requires_external_artifacts_when_enabled(tmp_path, monkeypatch) -> None:
    outputs_root = tmp_path / "outputs"
    minimal_files = [
        "outputs/index/index_summary.json",
        "outputs/index/label_cooccurrence.csv",
        "outputs/cache/cache_summary.json",
        "outputs/summary/runtime_preflight.json",
        "outputs/summary/model_selection.json",
        "outputs/summary/results_summary.json",
        "outputs/campaign/manifest.json",
        "outputs/campaign/stage_state.json",
        "outputs/checkpoints/resnet1d/best.pt",
        "outputs/predictions/resnet1d/raw/test_metrics_raw.json",
        "outputs/calibration/resnet1d/reports.json",
        "outputs/decision/resnet1d/isotonic/results.json",
        "outputs/robustness/resnet1d/results.json",
        "outputs/explanations/resnet1d/stability.csv",
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
        "outputs/tables/table_bootstrap_classification_ci.csv",
    ]
    for rel_path in minimal_files:
        _touch(tmp_path / rel_path)
    monkeypatch.setenv("RUN_BOOTSTRAP", "1")
    payload = validate_claim_artifacts(
        selected_model="resnet1d",
        require_external=True,
        external_enabled=False,
        outputs_root=outputs_root,
    )
    assert payload["internal_ptbxl_validation_passed"] is True
    assert payload["external_binary_validation_passed"] is False
    assert "outputs/external/external_manifest.parquet" in payload["missing_external_artifacts"]
    assert payload["external_five_superclass_validation_passed"] is False


def test_claim_run_script_includes_external_stages() -> None:
    script = Path("runpod/claim_run_ptbxl.sh").read_text(encoding="utf-8")
    assert "scripts/18_prepare_external_challenge2021.py" in script
    assert "scripts/19_eval_external.py" in script
    assert 'EXTERNAL_SOURCES="${EXTERNAL_SOURCES:-georgia,chapman_shaoxing}"' in script


def test_ptbxl_plus_exclusion_is_readme_only() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "PTB-XL Plus is intentionally excluded" in readme
    for root in [Path("ecg_reliability"), Path("scripts"), Path("runpod")]:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".sh", ".md", ".yaml"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            assert "PTB-XL Plus" not in text
