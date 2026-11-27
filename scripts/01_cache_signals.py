from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecg_reliability.data.cache import (
    CACHE_SCHEMA_VERSION,
    atomic_save_npy,
    atomic_write_json,
    atomic_write_table,
    compute_preprocessing_fingerprint,
    load_cache_provenance,
    source_record_fingerprint,
    validate_cached_signal,
)
from ecg_reliability.data.labels import TARGET_LABELS
from ecg_reliability.data.preprocessing import STANDARD_12_LEAD_ORDER, inspect_wfdb_header, load_ptbxl_signal
from ecg_reliability.io import read_table
from ecg_reliability.paths import resolve_path
from ecg_reliability.runtime import load_runtime_config


def cache_signals_from_index(
    config: dict[str, Any],
    *,
    overwrite: bool = False,
    summary_path: str | Path = "outputs/cache/cache_summary.json",
) -> dict[str, Any]:
    """Build or validate provenance-verified cached ECG arrays atomically."""
    data_cfg = config["data"]
    index_df = read_table(data_cfg["index_path"])
    ptbxl_root = Path(str(data_cfg["ptbxl_root"]))
    cache_dir = resolve_path(data_cfg["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(str(data_cfg["cache_manifest_path"]))
    provenance_path = manifest_path.parent / "cache_provenance.json"

    target_labels = list(data_cfg.get("target_labels", TARGET_LABELS))
    expected_lead_order = list(data_cfg.get("lead_order", STANDARD_12_LEAD_ORDER))
    target_sampling_rate = int(data_cfg["sampling_rate"])
    signal_length = int(data_cfg["signal_length"])
    num_leads = int(data_cfg["num_leads"])
    normalize = bool(data_cfg.get("normalize_per_lead", True))
    expected_shape = (num_leads, signal_length)
    fingerprint = compute_preprocessing_fingerprint(config)

    existing_provenance = load_cache_provenance(provenance_path) or {}
    existing_records: dict[str, dict[str, Any]] = existing_provenance.get("records", {})
    new_provenance_records: dict[str, dict[str, Any]] = {}

    rows: list[dict[str, Any]] = []
    failed_records: list[dict[str, Any]] = []
    example_resolved_paths: list[str] = []
    observed_shapes: list[tuple[int, ...]] = []
    observed_dtypes: list[str] = []
    observed_sampling_rates: list[float] = []
    observed_lead_orders: list[list[str]] = []
    observed_units: list[list[str]] = []
    invalidation_reasons: Counter[str] = Counter()
    n_reused_valid = 0
    n_rebuilt = 0
    nan_count = 0
    inf_count = 0

    for _, row in tqdm(index_df.iterrows(), total=len(index_df), desc="cache"):
        ecg_id = int(row["ecg_id"])
        ecg_key = str(ecg_id)
        cache_path = cache_dir / f"{ecg_id}.npy"
        record_path = ptbxl_root / str(row["record_path"])
        if len(example_resolved_paths) < 5:
            example_resolved_paths.append(str(record_path))
        try:
            header_info = inspect_wfdb_header(record_path)
            src_fp = source_record_fingerprint(record_path)
            is_valid = False
            reason = "overwrite_requested"
            if not overwrite:
                is_valid, reason = validate_cached_signal(
                    cache_path=cache_path,
                    expected_fingerprint=fingerprint,
                    expected_shape=expected_shape,
                    expected_lead_order=expected_lead_order,
                    expected_sampling_rate=float(target_sampling_rate),
                    expected_normalize=normalize,
                    expected_source_fingerprint=src_fp,
                    provenance_entry=existing_records.get(ecg_key),
                )

            if is_valid:
                n_reused_valid += 1
                signal = np.load(cache_path)
                new_provenance_records[ecg_key] = existing_records[ecg_key]
            else:
                invalidation_reasons[reason] += 1
                signal = load_ptbxl_signal(
                    record_path=record_path,
                    signal_length=signal_length,
                    normalize=normalize,
                    expected_sampling_rate=float(target_sampling_rate),
                    expected_lead_order=expected_lead_order,
                    enforce_lead_order=bool(data_cfg.get("enforce_lead_order", True)),
                    convert_to_mv=bool(data_cfg.get("convert_to_mv", True)),
                )
                atomic_save_npy(cache_path, signal.astype(np.float32))
                n_rebuilt += 1
                new_provenance_records[ecg_key] = {
                    "ecg_id": ecg_id,
                    "schema_version": CACHE_SCHEMA_VERSION,
                    "dataset_identity": str(data_cfg.get("dataset_identity", "ptb-xl-1.0.3")),
                    "preprocessing_fingerprint": fingerprint,
                    "source_fingerprint": src_fp,
                    "source_sampling_rate": header_info["sampling_rate"],
                    "target_sampling_rate": float(target_sampling_rate),
                    "source_lead_order": header_info["lead_order"],
                    "canonical_lead_order": expected_lead_order,
                    "source_units": header_info.get("units", []),
                    "amplitude_unit_after_preprocessing": "normalized_sd" if normalize else "mV",
                    "normalize_per_lead": normalize,
                    "shape": [int(dim) for dim in signal.shape],
                    "dtype": str(signal.dtype),
                    "cached_at_utc": datetime.now(timezone.utc).isoformat(),
                    "rebuild_reason": reason,
                }
        except Exception as exc:
            failed_records.append(
                {
                    "ecg_id": ecg_id,
                    "record_path": str(record_path),
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
            )
            continue

        observed_shapes.append(tuple(int(dim) for dim in signal.shape))
        observed_dtypes.append(str(signal.dtype))
        if header_info["sampling_rate"] is not None:
            observed_sampling_rates.append(float(header_info["sampling_rate"]))
        if header_info["lead_order"]:
            observed_lead_orders.append([str(name) for name in header_info["lead_order"]])
        if header_info.get("units"):
            observed_units.append([str(u) for u in header_info["units"]])
        nan_count += int(np.isnan(signal).sum())
        inf_count += int(np.isinf(signal).sum())

        manifest_row: dict[str, Any] = {
            "ecg_id": ecg_id,
            "cache_path": str(cache_path),
            "split": row["split"],
            "preprocessing_fingerprint": fingerprint,
            "amplitude_unit": "normalized_sd" if normalize else "mV",
            **{label: int(row[label]) for label in target_labels},
        }
        if "patient_id" in row.index and not pd.isna(row["patient_id"]):
            manifest_row["patient_id"] = int(row["patient_id"])
        rows.append(manifest_row)

    manifest = pd.DataFrame(rows)
    atomic_write_table(manifest, manifest_path)

    provenance_payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "dataset_identity": str(data_cfg.get("dataset_identity", "ptb-xl-1.0.3")),
        "preprocessing_fingerprint": fingerprint,
        "target_sampling_rate": target_sampling_rate,
        "signal_length": signal_length,
        "num_leads": num_leads,
        "canonical_lead_order": expected_lead_order,
        "normalize_per_lead": normalize,
        "amplitude_unit_after_preprocessing": "normalized_sd" if normalize else "mV",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "records": new_provenance_records,
    }
    atomic_write_json(provenance_path, provenance_payload)

    first_shape = list(observed_shapes[0]) if observed_shapes else None
    first_dtype = observed_dtypes[0] if observed_dtypes else None
    unique_lead_orders: list[list[str]] = []
    seen_lead_orders: set[tuple[str, ...]] = set()
    for lead_order in observed_lead_orders:
        key = tuple(lead_order)
        if key not in seen_lead_orders:
            seen_lead_orders.add(key)
            unique_lead_orders.append(lead_order)
    unique_units: list[list[str]] = []
    seen_units: set[tuple[str, ...]] = set()
    for unit_list in observed_units:
        key = tuple(unit_list)
        if key not in seen_units:
            seen_units.add(key)
            unique_units.append(unit_list)

    unique_sampling_rates = sorted({float(rate) for rate in observed_sampling_rates})
    expected_shape_list = [num_leads, signal_length]
    warning_lead_order_mismatch = len(unique_lead_orders) > 1
    warning_sampling_rate_mismatch = any(abs(rate - float(target_sampling_rate)) > 1e-6 for rate in unique_sampling_rates)
    warning_signal_shape_mismatch = any(list(shape) != expected_shape_list for shape in observed_shapes)
    summary = {
        "status": "real_ptbxl_cache_prepared",
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "dataset_identity": str(data_cfg.get("dataset_identity", "ptb-xl-1.0.3")),
        "preprocessing_fingerprint": fingerprint,
        "provenance_manifest_path": str(provenance_path),
        "n_index_records": int(len(index_df)),
        "n_cached": int(len(manifest)),
        "n_reused_valid": int(n_reused_valid),
        "n_rebuilt": int(n_rebuilt),
        "invalidation_reasons": dict(invalidation_reasons),
        "n_failed": int(len(failed_records)),
        "failed_records": failed_records,
        "example_resolved_record_paths": example_resolved_paths,
        "observed_signal_shape": first_shape,
        "signal_shape_observed": first_shape,
        "expected_signal_shape": expected_shape_list,
        "dtype": first_dtype,
        "nan_count": int(nan_count),
        "inf_count": int(inf_count),
        "observed_sampling_rates": unique_sampling_rates,
        "target_sampling_rate": target_sampling_rate,
        "observed_lead_orders": unique_lead_orders,
        "canonical_lead_order": expected_lead_order,
        "observed_physical_units": unique_units,
        "amplitude_unit_after_preprocessing": "normalized_sd" if normalize else "mV",
        "warning_lead_order_mismatch": warning_lead_order_mismatch,
        "warning_sampling_rate_not_100": warning_sampling_rate_mismatch,
        "warning_signal_shape_mismatch": warning_signal_shape_mismatch,
        "cache_dir": str(cache_dir),
        "cache_manifest_path": str(manifest_path),
        "overwrite": bool(overwrite),
    }
    atomic_write_json(summary_path, summary)
    if len(manifest) == 0:
        raise RuntimeError(f"Signal caching failed for every record. See {summary_path}.")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_runtime_config(args.config, args.override)
    summary = cache_signals_from_index(config, overwrite=bool(args.overwrite))
    print(
        f"Cached {summary['n_cached']} signals to {summary['cache_dir']} "
        f"(reused={summary['n_reused_valid']}, rebuilt={summary['n_rebuilt']}, failed={summary['n_failed']})"
    )
    print("Cache summary: outputs/cache/cache_summary.json")


if __name__ == "__main__":
    main()
