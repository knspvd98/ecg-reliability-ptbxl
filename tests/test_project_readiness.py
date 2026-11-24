from __future__ import annotations

import pandas as pd
import torch

from ecg_reliability.data.preprocessing import inspect_wfdb_header
from ecg_reliability.data.ptbxl_index import build_index_summary, compute_patient_overlap_report, label_cooccurrence
from ecg_reliability.training.checkpoints import load_checkpoint, save_checkpoint


def test_index_summary_and_cooccurrence_generation() -> None:
    index_df = pd.DataFrame(
        {
            "ecg_id": [1, 2, 3],
            "patient_id": [10, 11, 12],
            "strat_fold": [1, 9, 10],
            "split": ["train", "val", "test"],
            "record_path": ["a", "b", "c"],
            "NORM": [1, 0, 0],
            "MI": [0, 1, 0],
            "STTC": [0, 0, 1],
            "CD": [0, 0, 0],
            "HYP": [0, 1, 0],
        }
    )
    config = {
        "data": {
            "filter_empty_targets": True,
            "num_leads": 12,
            "signal_length": 1000,
            "ptbxl_root": "/tmp/ptbxl",
            "index_path": "outputs/index/ptbxl_index.parquet",
        }
    }
    overlap = compute_patient_overlap_report(index_df)
    summary = build_index_summary(
        index_df,
        config,
        empty_target_count_before_filter=0,
        overlap_report=overlap,
        metadata={
            "raw_ptbxl_row_count": 4,
            "required_ptbxl_columns_present": ["ecg_id", "patient_id", "scp_codes", "strat_fold", "filename_lr", "filename_hr"],
            "required_ptbxl_columns_missing": [],
            "filename_lr_present": True,
            "filename_hr_present": True,
            "scp_codes_present": True,
            "scp_statements_mapping_succeeded": True,
        },
    )
    cooc = label_cooccurrence(index_df)
    assert summary["status"] == "real_ptbxl_index_prepared"
    assert summary["expected_ptbxl_version"] == "1.0.3"
    assert summary["raw_ptbxl_row_count"] == 4
    assert summary["post_filter_task_row_count"] == 3
    assert summary["total_records"] == 3
    assert summary["split_counts"]["train"] == 1
    assert summary["patient_overlap_train_val"] == 0
    assert summary["scp_statements_mapping_succeeded"] is True
    assert cooc.loc["MI", "HYP"] == 1


def test_compute_patient_overlap_report_handles_missing_patient_id() -> None:
    frame = pd.DataFrame({"split": ["train", "val"], "NORM": [1, 0], "MI": [0, 1], "STTC": [0, 0], "CD": [0, 0], "HYP": [0, 0]})
    report = compute_patient_overlap_report(frame)
    assert report["patient_leakage_checked"] is False
    assert report["patient_overlap_train_val"] is None


def test_load_checkpoint_roundtrip(tmp_path) -> None:
    checkpoint_path = tmp_path / "roundtrip.pt"
    payload = {"epoch": 1, "history": [{"loss": 0.1}], "tensor": torch.tensor([1.0, 2.0])}
    save_checkpoint(checkpoint_path, payload)
    loaded = load_checkpoint(checkpoint_path)
    assert loaded["epoch"] == 1
    assert torch.equal(loaded["tensor"], payload["tensor"])


def test_inspect_wfdb_header_uses_observed_header_fields(monkeypatch) -> None:
    class FakeHeader:
        fs = 100
        sig_name = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        sig_len = 1000
        n_sig = 12

    class FakeWFDB:
        @staticmethod
        def rdheader(path: str):
            return FakeHeader()

    monkeypatch.setattr("ecg_reliability.data.preprocessing._require_wfdb", lambda: FakeWFDB())
    report = inspect_wfdb_header("records100/00001_hr")
    assert report["sampling_rate"] == 100.0
    assert report["signal_length"] == 1000
    assert report["num_leads"] == 12
    assert report["lead_order"][0] == "I"
