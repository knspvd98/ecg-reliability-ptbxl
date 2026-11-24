from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ecg_reliability.external.cache_external import standardize_external_signal
from ecg_reliability.external.challenge2021 import (
    STANDARD_12_LEAD_ORDER,
    build_challenge2021_manifest,
    resolve_source_directory,
)
from ecg_reliability.external.evaluate_external import evaluate_external_model
from ecg_reliability.external.label_mapping import binary_abnormal_label, parse_snomed_dx_codes


def _write_fake_header(path: Path, *, dx_codes: str) -> None:
    lines = ["rec001 12 500 5000"]
    leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    for lead in leads:
        lines.append(f"rec001.dat 16 1000/mV 0 0 0 0 {lead}")
    lines.extend(
        [
            "#Age: 45",
            "#Sex: Female",
            f"#Dx: {dx_codes}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_parse_snomed_dx_codes_and_binary_mapping() -> None:
    comments = ["#Age: 34", "#Dx: 426783006,164889003"]
    dx_codes = parse_snomed_dx_codes(comments)
    assert dx_codes == ["426783006", "164889003"]
    assert binary_abnormal_label(dx_codes, ["426783006"]) == 1
    assert binary_abnormal_label(["426783006"], ["426783006"]) == 0
    assert binary_abnormal_label([], ["426783006"]) is None


def test_build_external_manifest_from_direct_source_directory(tmp_path) -> None:
    source_dir = tmp_path / "georgia_direct"
    source_dir.mkdir(parents=True, exist_ok=True)
    _write_fake_header(source_dir / "rec001.hea", dx_codes="426783006")
    manifest = build_challenge2021_manifest(
        root=None,
        sources=["georgia"],
        normal_snomed_codes=["426783006"],
        allow_sources={"georgia": True},
        max_records_per_source=None,
        source_roots={"georgia": str(source_dir)},
    )
    assert len(manifest) == 1
    row = manifest.iloc[0]
    assert row["source"] == "georgia"
    assert row["binary_abnormal_label"] == 0
    assert row["status"] == "ready"
    assert row["source_root_type"] == "direct_source_root"
    assert row["lead_names_original"][3] == "AVR"


def test_resolve_source_directory_falls_back_to_nested_root(tmp_path) -> None:
    source_dir = tmp_path / "training" / "chapman_shaoxing"
    source_dir.mkdir(parents=True, exist_ok=True)
    _write_fake_header(source_dir / "rec001.hea", dx_codes="426783006")
    resolved, root_type = resolve_source_directory(
        "chapman_shaoxing",
        source_roots={},
        nested_root=tmp_path,
    )
    assert resolved == source_dir
    assert root_type == "nested_challenge_root"


def test_standardize_external_signal_resamples_and_reorders(monkeypatch) -> None:
    class FakeWFDB:
        @staticmethod
        def rdsamp(path: str):
            signal = np.random.default_rng(0).normal(size=(5000, 12)).astype(np.float32)
            fields = {
                "fs": 500,
                "sig_name": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
            }
            return signal, fields

    monkeypatch.setattr("ecg_reliability.external.cache_external._require_wfdb", lambda: FakeWFDB())
    signal, metadata = standardize_external_signal(
        "fake_record",
        expected_lead_order=STANDARD_12_LEAD_ORDER,
        target_sampling_rate=100,
        signal_length=1000,
        normalize=True,
    )
    assert signal.shape == (12, 1000)
    assert signal.dtype == np.float32
    assert metadata["sampling_rate_original"] == 500
    assert metadata["lead_order_original"][3] == "AVR"


def test_evaluate_external_model_binary_results_do_not_claim_five_superclass(monkeypatch, tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "external_record_id": ["a", "b"],
            "source": ["georgia", "chapman_shaoxing"],
            "cache_path": [str(tmp_path / "a.npy"), str(tmp_path / "b.npy")],
            "binary_abnormal_label": [0, 1],
            "status": ["cached", "cached"],
            "dx_codes": [["426783006"], ["164889003"]],
        }
    )

    monkeypatch.setattr("ecg_reliability.external.evaluate_external._load_external_frame", lambda config: frame.copy())
    monkeypatch.setattr(
        "ecg_reliability.external.evaluate_external._predict_torch_logits",
        lambda frame, model_name, config: (
            np.asarray(
                [
                    [4.0, -4.0, -4.0, -4.0, -4.0],
                    [-4.0, 4.0, -2.0, -2.0, -2.0],
                ],
                dtype=np.float32,
            ),
            __import__("torch").device("cpu"),
        ),
    )
    config = {
        "training": {"device": "cpu", "eval_batch_size": 2, "non_blocking_transfer": False},
        "external": {"manifest_path": "unused", "snomed_overlap_enabled": False},
    }
    results, scored_frame, sourcewise, overlap, runtime = evaluate_external_model(
        model_name="resnet1d",
        config=config,
        calibration_method=None,
    )
    assert results["validation_task"] == "external_binary_abnormal_vs_normal"
    assert results["external_five_superclass_validation"] is False
    assert results["primary_external_score"] == "p_abnormal_max"
    assert results["raw_only"] is True
    assert "five-superclass" in results["warning"]
    assert list(sourcewise["source"]) == ["chapman_shaoxing", "georgia"]
    assert overlap is None
    assert runtime["device"] == "cpu"
    assert "p_abnormal_max" in scored_frame.columns
