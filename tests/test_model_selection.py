from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from ecg_reliability.evaluation.model_selection import normalize_deep_model_candidates, select_best_deep_model


def test_model_selection_excludes_non_deep_baselines_by_default() -> None:
    candidates = normalize_deep_model_candidates(["xgboost", "simple_cnn", "resnet1d", "tcn", "transformer"])
    assert candidates == ["resnet1d", "tcn", "transformer"]


def test_model_selection_writes_selected_deep_model(tmp_path) -> None:
    prediction_root = tmp_path / "outputs" / "predictions"
    for model_name, val_auroc, val_auprc in [
        ("resnet1d", 0.80, 0.50),
        ("inceptiontime", 0.84, 0.48),
        ("tcn", 0.79, 0.60),
    ]:
        model_dir = prediction_root / model_name / "raw"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "val_metrics_raw.json").write_text(
            json.dumps({"macro_auroc": val_auroc, "macro_auprc": val_auprc}),
            encoding="utf-8",
        )

    payload = select_best_deep_model(prediction_root, ["resnet1d", "inceptiontime", "tcn"])
    assert payload["selected_deep_model"] == "inceptiontime"
    assert "available_metrics" in payload


def test_claim_bearing_figure_script_requires_model_or_manifest(tmp_path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    missing_manifest = tmp_path / "missing_selection.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/09_build_figures.py",
            "--config",
            "configs/debug.yaml",
            "--selection-manifest",
            str(missing_manifest),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Claim-bearing analysis requires --model-name or a selection manifest" in (result.stderr + result.stdout)


def test_claim_bearing_summary_script_requires_model_or_manifest(tmp_path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    missing_manifest = tmp_path / "missing_selection.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/10_build_results_summary.py",
            "--config",
            "configs/debug.yaml",
            "--selection-manifest",
            str(missing_manifest),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Claim-bearing analysis requires --model-name or a selection manifest" in (result.stderr + result.stdout)
