from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ecg_reliability.data.labels import TARGET_LABELS, add_target_columns
from ecg_reliability.io import write_table

REQUIRED_PTBXL_COLUMNS = [
    "ecg_id",
    "patient_id",
    "scp_codes",
    "strat_fold",
    "filename_lr",
    "filename_hr",
]


def _assign_split(fold: Any, train_folds: set[int], val_fold: int, test_fold: int) -> str | None:
    if pd.isna(fold):
        return None
    fold_int = int(fold)
    if fold_int in train_folds:
        return "train"
    if fold_int == val_fold:
        return "val"
    if fold_int == test_fold:
        return "test"
    return None


def compute_patient_overlap_report(index_df: pd.DataFrame) -> dict[str, Any]:
    """Verify patient-level disjointness across train, validation, and test splits."""
    if "patient_id" not in index_df.columns or index_df["patient_id"].isna().all():
        return {
            "patient_leakage_checked": False,
            "patient_overlap_train_val": None,
            "patient_overlap_train_test": None,
            "patient_overlap_val_test": None,
            "has_patient_leakage": None,
            "missing_patient_id_count": int(len(index_df)) if "patient_id" not in index_df.columns else int(index_df["patient_id"].isna().sum()),
            "unique_patients_per_split": {},
        }

    missing_patient_ids = int(index_df["patient_id"].isna().sum())
    valid = index_df.loc[index_df["patient_id"].notna()]
    split_patients: dict[str, set[Any]] = {}
    for split in ("train", "val", "test"):
        split_patients[split] = set(valid.loc[valid["split"] == split, "patient_id"].tolist())

    overlap_train_val = len(split_patients["train"] & split_patients["val"])
    overlap_train_test = len(split_patients["train"] & split_patients["test"])
    overlap_val_test = len(split_patients["val"] & split_patients["test"])
    has_leakage = bool((overlap_train_val + overlap_train_test + overlap_val_test) > 0)
    return {
        "patient_leakage_checked": True,
        "patient_overlap_train_val": int(overlap_train_val),
        "patient_overlap_train_test": int(overlap_train_test),
        "patient_overlap_val_test": int(overlap_val_test),
        "has_patient_leakage": has_leakage,
        "missing_patient_id_count": missing_patient_ids,
        "unique_patients_per_split": {split: int(len(patients)) for split, patients in split_patients.items()},
    }


def label_cooccurrence(index_df: pd.DataFrame, label_names: list[str] | None = None) -> pd.DataFrame:
    """Compute symmetric diagnostic superclass co-occurrence count matrix."""
    labels = list(label_names or TARGET_LABELS)
    missing = [label for label in labels if label not in index_df.columns]
    if missing:
        raise ValueError(f"Missing label columns for co-occurrence calculation: {missing}")
    matrix = index_df[labels].astype(int).to_numpy()
    cooc = matrix.T @ matrix
    return pd.DataFrame(cooc, index=labels, columns=labels)


def prepare_ptbxl_index(config: dict[str, Any], *, strict_patient_disjoint: bool = True) -> pd.DataFrame:
    """Load PTB-XL v1.0.3 metadata, harmonize diagnostic superclasses, and enforce official folds."""
    data_cfg = config["data"]
    ptbxl_root = Path(str(data_cfg["ptbxl_root"]))
    database_csv = ptbxl_root / "ptbxl_database.csv"
    statements_csv = ptbxl_root / "scp_statements.csv"
    if not database_csv.exists():
        raise FileNotFoundError(f"Missing PTB-XL database metadata file: {database_csv}")
    if not statements_csv.exists():
        raise FileNotFoundError(f"Missing PTB-XL SCP statements file: {statements_csv}")

    raw_df = pd.read_csv(database_csv)
    raw_count = int(len(raw_df))
    present_cols = [col for col in REQUIRED_PTBXL_COLUMNS if col in raw_df.columns]
    missing_cols = [col for col in REQUIRED_PTBXL_COLUMNS if col not in raw_df.columns]
    if missing_cols:
        raise ValueError(f"PTB-XL metadata is missing required columns: {missing_cols}")

    if raw_df["ecg_id"].duplicated().any():
        dup_count = int(raw_df["ecg_id"].duplicated().sum())
        raise ValueError(f"PTB-XL metadata contains {dup_count} duplicate ecg_id rows.")

    scp_statements = pd.read_csv(statements_csv, index_col=0)
    filter_empty = bool(data_cfg.get("filter_empty_targets", True))
    target_labels = list(data_cfg.get("target_labels", TARGET_LABELS))
    min_likelihood = float(data_cfg.get("min_scp_likelihood", 0.0))

    labeled_df = add_target_columns(
        raw_df,
        scp_statements,
        filter_empty_targets=filter_empty,
        target_labels=target_labels,
        min_likelihood=min_likelihood,
    )
    empty_before_filter = int(labeled_df.attrs.get("empty_target_count_before_filter", 0))

    train_folds = {int(f) for f in data_cfg.get("train_folds", [1, 2, 3, 4, 5, 6, 7, 8])}
    val_fold = int(data_cfg.get("val_fold", 9))
    test_fold = int(data_cfg.get("test_fold", 10))
    if val_fold in train_folds or test_fold in train_folds or val_fold == test_fold:
        raise ValueError(
            f"Invalid overlapping PTB-XL fold configuration: train={sorted(train_folds)}, val={val_fold}, test={test_fold}"
        )

    labeled_df["split"] = [
        _assign_split(fold, train_folds=train_folds, val_fold=val_fold, test_fold=test_fold)
        for fold in labeled_df["strat_fold"]
    ]
    unassigned_count = int(labeled_df["split"].isna().sum())
    labeled_df = labeled_df.loc[labeled_df["split"].notna()].reset_index(drop=True)

    sampling_rate = int(data_cfg.get("sampling_rate", 100))
    record_col = "filename_hr" if sampling_rate >= 500 else "filename_lr"
    labeled_df["record_path"] = labeled_df[record_col].astype(str)

    debug_max = data_cfg.get("debug_max_records")
    if isinstance(debug_max, int) and debug_max > 0:
        # Preserve stratified split presence without breaking patient grouping.
        parts = []
        for split_name in ("train", "val", "test"):
            split_df = labeled_df.loc[labeled_df["split"] == split_name]
            if split_df.empty:
                continue
            parts.append(split_df.head(debug_max))
        labeled_df = pd.concat(parts, axis=0, ignore_index=True)

    overlap_report = compute_patient_overlap_report(labeled_df)
    if strict_patient_disjoint and overlap_report.get("has_patient_leakage"):
        raise ValueError(
            "Patient leakage detected across PTB-XL splits: "
            f"train∩val={overlap_report['patient_overlap_train_val']}, "
            f"train∩test={overlap_report['patient_overlap_train_test']}, "
            f"val∩test={overlap_report['patient_overlap_val_test']}"
        )

    labeled_df.attrs["ptbxl_prepare_metadata"] = {
        "raw_ptbxl_row_count": raw_count,
        "empty_target_count_before_filter": empty_before_filter,
        "unassigned_fold_count": unassigned_count,
        "required_ptbxl_columns_present": present_cols,
        "required_ptbxl_columns_missing": missing_cols,
        "filename_lr_present": "filename_lr" in raw_df.columns,
        "filename_hr_present": "filename_hr" in raw_df.columns,
        "scp_codes_present": "scp_codes" in raw_df.columns,
        "scp_statements_mapping_succeeded": True,
        "record_path_column_used": record_col,
        "overlap_report": overlap_report,
    }
    return labeled_df


def save_prepared_index(index_df: pd.DataFrame, path: str | Path) -> None:
    write_table(index_df, path)


def build_index_summary(
    index_df: pd.DataFrame,
    config: dict[str, Any],
    *,
    empty_target_count_before_filter: int | None = None,
    overlap_report: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build machine-readable summary and provenance report for the prepared PTB-XL index."""
    data_cfg = config.get("data", {})
    labels = list(data_cfg.get("target_labels", TARGET_LABELS))
    metadata = metadata or {}
    overlap = overlap_report if overlap_report is not None else compute_patient_overlap_report(index_df)

    split_counts = {
        split: int((index_df["split"] == split).sum())
        for split in ("train", "val", "test")
    }
    label_counts = {
        label: int(index_df[label].sum()) if label in index_df.columns else 0
        for label in labels
    }
    label_prevalence = {
        label: float(index_df[label].mean()) if label in index_df.columns and len(index_df) > 0 else 0.0
        for label in labels
    }

    return {
        "status": "real_ptbxl_index_prepared",
        "expected_ptbxl_version": "1.0.3",
        "ptbxl_root": str(data_cfg.get("ptbxl_root", "")),
        "index_path": str(data_cfg.get("index_path", "")),
        "raw_ptbxl_row_count": metadata.get("raw_ptbxl_row_count", int(len(index_df))),
        "post_filter_task_row_count": int(len(index_df)),
        "total_records": int(len(index_df)),
        "empty_target_count_before_filter": empty_target_count_before_filter,
        "filter_empty_targets": bool(data_cfg.get("filter_empty_targets", True)),
        "split_counts": split_counts,
        "label_counts": label_counts,
        "label_prevalence": label_prevalence,
        "num_leads": int(data_cfg.get("num_leads", 12)),
        "signal_length": int(data_cfg.get("signal_length", 1000)),
        "sampling_rate": int(data_cfg.get("sampling_rate", 100)),
        "required_ptbxl_columns_present": metadata.get("required_ptbxl_columns_present", []),
        "required_ptbxl_columns_missing": metadata.get("required_ptbxl_columns_missing", []),
        "filename_lr_present": metadata.get("filename_lr_present"),
        "filename_hr_present": metadata.get("filename_hr_present"),
        "scp_codes_present": metadata.get("scp_codes_present"),
        "scp_statements_mapping_succeeded": bool(metadata.get("scp_statements_mapping_succeeded", False)),
        "patient_leakage_checked": overlap.get("patient_leakage_checked", False),
        "patient_overlap_train_val": overlap.get("patient_overlap_train_val"),
        "patient_overlap_train_test": overlap.get("patient_overlap_train_test"),
        "patient_overlap_val_test": overlap.get("patient_overlap_val_test"),
        "has_patient_leakage": overlap.get("has_patient_leakage"),
        "unique_patients_per_split": overlap.get("unique_patients_per_split", {}),
    }
