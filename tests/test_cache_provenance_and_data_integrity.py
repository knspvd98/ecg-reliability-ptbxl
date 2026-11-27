from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ecg_reliability.config import validate_config
from ecg_reliability.data.cache import compute_preprocessing_fingerprint, validate_cached_signal
from ecg_reliability.data.dataset import CachedECGDataset
from ecg_reliability.data.features import compute_signal_quality_indicators
from ecg_reliability.data.labels import parse_scp_codes
from ecg_reliability.data.preprocessing import (
    STANDARD_12_LEAD_ORDER,
    load_ptbxl_signal,
    normalize_per_lead,
    reorder_leads,
)
from ecg_reliability.data.ptbxl_index import compute_patient_overlap_report, prepare_ptbxl_index


def _load_cache_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "01_cache_signals.py"
    spec = importlib.util.spec_from_file_location("cache_signals_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_scp_codes_rejects_malformed_data() -> None:
    with pytest.raises(ValueError, match="Malformed scp_codes"):
        parse_scp_codes("{not valid python")
    with pytest.raises(ValueError, match="Expected dict"):
        parse_scp_codes("['NORM']")
    with pytest.raises(ValueError, match="Non-numeric likelihood"):
        parse_scp_codes("{'NORM': 'high'}")


def test_reorder_leads_fixes_scrambled_order_and_rejects_missing_or_duplicate() -> None:
    scrambled = list(reversed(STANDARD_12_LEAD_ORDER))
    sig = np.arange(12 * 10, dtype=np.float32).reshape(12, 10)
    reordered = reorder_leads(sig, scrambled, STANDARD_12_LEAD_ORDER)
    assert np.array_equal(reordered[0], sig[11])
    assert np.array_equal(reordered[11], sig[0])

    with pytest.raises(ValueError, match="Missing required ECG leads"):
        reorder_leads(sig[:11], STANDARD_12_LEAD_ORDER[:11], STANDARD_12_LEAD_ORDER)

    dup_names = ["I", "I", *STANDARD_12_LEAD_ORDER[2:]]
    with pytest.raises(ValueError, match="Duplicate lead names"):
        reorder_leads(sig, dup_names, STANDARD_12_LEAD_ORDER)


def test_load_ptbxl_signal_converts_units_and_rejects_non_finite(monkeypatch) -> None:
    class FakeWFDBMicrovolt:
        @staticmethod
        def rdsamp(path: str):
            data = np.ones((1000, 12), dtype=np.float32) * 1000.0  # 1000 uV = 1 mV
            data[0, :] = 2000.0
            fields = {
                "fs": 100,
                "sig_name": list(reversed(STANDARD_12_LEAD_ORDER)),
                "units": ["uV"] * 12,
            }
            return data, fields

    monkeypatch.setattr("ecg_reliability.data.preprocessing._require_wfdb", lambda: FakeWFDBMicrovolt())
    arr = load_ptbxl_signal("dummy", signal_length=1000, normalize=False, expected_sampling_rate=100)
    assert arr.shape == (12, 1000)
    assert np.isclose(arr[0, 0], 2.0)
    assert np.isclose(arr[0, 1], 1.0)

    class FakeWFDBNaN:
        @staticmethod
        def rdsamp(path: str):
            data = np.zeros((1000, 12), dtype=np.float32)
            data[5, 2] = np.nan
            return data, {"fs": 100, "sig_name": STANDARD_12_LEAD_ORDER, "units": ["mV"] * 12}

    monkeypatch.setattr("ecg_reliability.data.preprocessing._require_wfdb", lambda: FakeWFDBNaN())
    with pytest.raises(ValueError, match="non-finite"):
        load_ptbxl_signal("dummy")


def test_patient_leakage_detected_and_rejected(tmp_path) -> None:
    ptbxl_root = tmp_path / "ptbxl"
    ptbxl_root.mkdir()
    pd.DataFrame(
        {
            "ecg_id": [1, 2, 3],
            "patient_id": [42, 42, 99],  # Patient 42 leaked between train (fold 1) and val (fold 9)
            "scp_codes": ["{'NORM': 100}", "{'NORM': 100}", "{'IMI': 100}"],
            "strat_fold": [1, 9, 10],
            "filename_lr": ["r1", "r2", "r3"],
            "filename_hr": ["r1", "r2", "r3"],
        }
    ).to_csv(ptbxl_root / "ptbxl_database.csv", index=False)
    pd.DataFrame(
        {"diagnostic": [1, 1], "diagnostic_class": ["NORM", "MI"]},
        index=["NORM", "IMI"],
    ).to_csv(ptbxl_root / "scp_statements.csv")

    config = {
        "data": {
            "ptbxl_root": str(ptbxl_root),
            "sampling_rate": 100,
            "signal_length": 1000,
            "num_leads": 12,
            "train_folds": [1, 2, 3, 4, 5, 6, 7, 8],
            "val_fold": 9,
            "test_fold": 10,
            "filter_empty_targets": True,
        }
    }
    with pytest.raises(ValueError, match="Patient leakage detected"):
        prepare_ptbxl_index(config, strict_patient_disjoint=True)

    df = prepare_ptbxl_index(config, strict_patient_disjoint=False)
    report = compute_patient_overlap_report(df)
    assert report["has_patient_leakage"] is True
    assert report["patient_overlap_train_val"] == 1


def test_provenance_aware_cache_reuses_valid_and_rebuilds_stale(tmp_path, monkeypatch) -> None:
    cache_mod = _load_cache_script()
    ptbxl_root = tmp_path / "ptbxl"
    ptbxl_root.mkdir()
    rec_base = ptbxl_root / "rec1"
    (rec_base.with_suffix(".hea")).write_text("rec1 12 100 1000\n", encoding="utf-8")
    (rec_base.with_suffix(".dat")).write_bytes(b"\x00" * 16)

    index_path = tmp_path / "index.csv"
    pd.DataFrame(
        [{"ecg_id": 1, "patient_id": 10, "record_path": "rec1", "split": "train", "NORM": 1, "MI": 0, "STTC": 0, "CD": 0, "HYP": 0}]
    ).to_csv(index_path, index=False)

    load_calls = {"count": 0}

    def fake_header(path):
        return {
            "sampling_rate": 100.0,
            "signal_length": 1000,
            "num_leads": 12,
            "lead_order": STANDARD_12_LEAD_ORDER,
            "canonical_lead_order": STANDARD_12_LEAD_ORDER,
            "units": ["mV"] * 12,
        }

    def fake_load(**kwargs):
        load_calls["count"] += 1
        return np.ones((12, 1000), dtype=np.float32) * (2.0 if not kwargs.get("normalize") else 0.0)

    monkeypatch.setattr(cache_mod, "inspect_wfdb_header", fake_header)
    monkeypatch.setattr(cache_mod, "load_ptbxl_signal", fake_load)

    config = {
        "data": {
            "ptbxl_root": str(ptbxl_root),
            "index_path": str(index_path),
            "cache_dir": str(tmp_path / "signals"),
            "cache_manifest_path": str(tmp_path / "cache_manifest.csv"),
            "sampling_rate": 100,
            "signal_length": 1000,
            "num_leads": 12,
            "normalize_per_lead": True,
            "target_labels": ["NORM", "MI", "STTC", "CD", "HYP"],
        }
    }
    first = cache_mod.cache_signals_from_index(config, summary_path=tmp_path / "summary.json")
    assert first["n_rebuilt"] == 1
    assert first["n_reused_valid"] == 0
    assert load_calls["count"] == 1

    # Second run with identical config & source must reuse without reloading
    second = cache_mod.cache_signals_from_index(config, summary_path=tmp_path / "summary.json")
    assert second["n_reused_valid"] == 1
    assert second["n_rebuilt"] == 0
    assert load_calls["count"] == 1

    # Changing normalization invalidates provenance and forces rebuild (no existence-only reuse!)
    config["data"]["normalize_per_lead"] = False
    third = cache_mod.cache_signals_from_index(config, summary_path=tmp_path / "summary.json")
    assert third["n_rebuilt"] == 1
    assert third["n_reused_valid"] == 0
    assert "preprocessing_fingerprint_mismatch" in third["invalidation_reasons"]
    assert load_calls["count"] == 2

    # Direct validation helper catches missing provenance on pre-existing .npy
    orphan = tmp_path / "orphan.npy"
    np.save(orphan, np.zeros((12, 1000), dtype=np.float32))
    ok, reason = validate_cached_signal(
        cache_path=orphan,
        expected_fingerprint=compute_preprocessing_fingerprint(config),
        expected_shape=(12, 1000),
        expected_lead_order=STANDARD_12_LEAD_ORDER,
        expected_sampling_rate=100.0,
        expected_normalize=False,
        expected_source_fingerprint={},
        provenance_entry=None,
    )
    assert ok is False
    assert reason == "missing_provenance_entry"


def test_dataset_rejects_missing_artifact_and_nan_values(tmp_path) -> None:
    bad_path = tmp_path / "bad.npy"
    arr = np.zeros((12, 1000), dtype=np.float32)
    arr[0, 5] = np.nan
    np.save(bad_path, arr)
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {"ecg_id": 1, "cache_path": str(bad_path), "split": "test", "NORM": 1, "MI": 0, "STTC": 0, "CD": 0, "HYP": 0},
            {"ecg_id": 2, "cache_path": str(tmp_path / "missing.npy"), "split": "test", "NORM": 1, "MI": 0, "STTC": 0, "CD": 0, "HYP": 0},
        ]
    ).to_csv(manifest, index=False)

    ds = CachedECGDataset(manifest, split="test")
    with pytest.raises(ValueError, match="NaN or Inf"):
        _ = ds[0]
    with pytest.raises(FileNotFoundError, match="Missing cached ECG"):
        _ = ds[1]

    with pytest.raises(ValueError, match="NaN or Inf"):
        normalize_per_lead(arr)


def test_sqi_detects_flatline_and_config_validator_rejects_overlap() -> None:
    sig = np.random.default_rng(7).normal(0, 1, size=(12, 1000)).astype(np.float32)
    sig[3, :] = 0.0
    sqi = compute_signal_quality_indicators(sig, sampling_rate=100)
    assert sqi["n_flatline_leads"] == 1
    assert "aVR" in sqi["flatline_leads"]

    with pytest.raises(ValueError, match="Overlapping PTB-XL folds"):
        validate_config({"data": {"train_folds": [1, 9], "val_fold": 9, "test_fold": 10}})
