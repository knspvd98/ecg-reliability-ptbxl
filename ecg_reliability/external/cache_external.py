from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ecg_reliability.data.preprocessing import normalize_per_lead, pad_or_trim_signal
from ecg_reliability.external.challenge2021 import STANDARD_12_LEAD_ORDER, canonical_lead_name


def _require_wfdb():
    try:
        import wfdb
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("wfdb is required to cache external signals.") from exc
    return wfdb


def _resample_signal(signal: np.ndarray, original_fs: float, target_fs: int) -> np.ndarray:
    if abs(float(original_fs) - float(target_fs)) < 1e-6:
        return signal
    from scipy.signal import resample

    target_length = max(1, int(round(signal.shape[1] * float(target_fs) / float(original_fs))))
    return resample(signal, target_length, axis=1).astype(np.float32)


def _lead_index_map(lead_names: list[str]) -> dict[str, int]:
    return {canonical_lead_name(name): idx for idx, name in enumerate(lead_names)}


def standardize_external_signal(
    record_path: str | Path,
    *,
    expected_lead_order: list[str],
    target_sampling_rate: int,
    signal_length: int,
    normalize: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    wfdb = _require_wfdb()
    signal, fields = wfdb.rdsamp(str(record_path))
    original_signal = signal.T.astype(np.float32)
    lead_names = [canonical_lead_name(name) for name in list(fields.get("sig_name", []))]
    index_map = _lead_index_map(lead_names)
    missing_leads = [lead for lead in expected_lead_order if lead not in index_map]
    if missing_leads:
        raise ValueError(f"Missing required leads: {missing_leads}")
    ordered = np.stack([original_signal[index_map[lead]] for lead in expected_lead_order], axis=0)
    original_fs = float(fields.get("fs", 0.0) or 0.0)
    if original_fs <= 0:
        raise ValueError("Invalid sampling frequency in external record.")
    standardized = _resample_signal(ordered, original_fs, target_sampling_rate)
    standardized = pad_or_trim_signal(standardized, target_length=signal_length)
    if normalize:
        standardized = normalize_per_lead(standardized)
    standardized = standardized.astype(np.float32)
    metadata = {
        "sampling_rate_original": original_fs,
        "lead_order_original": lead_names,
        "observed_signal_shape": [int(dim) for dim in standardized.shape],
    }
    return standardized, metadata


def _safe_cache_name(external_record_id: str) -> str:
    return external_record_id.replace("/", "__").replace(":", "__")


def cache_external_manifest(manifest: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    external_cfg = config["external"]
    cache_root = Path(external_cfg["cache_dir"])
    cache_root.mkdir(parents=True, exist_ok=True)
    updated = manifest.copy()
    failed_records: list[dict[str, Any]] = []
    warnings: list[str] = []
    observed_sampling_rates: set[float] = set()
    observed_lead_orders: list[list[str]] = []
    observed_shapes: list[list[int]] = []
    nan_count = 0
    inf_count = 0

    if updated.empty:
        summary = {
            "enabled": bool(external_cfg.get("enabled", False)),
            "root": external_cfg.get("root"),
            "sources_requested": list(external_cfg.get("sources", [])),
            "sources_found": [],
            "n_records_found": 0,
            "n_cached": 0,
            "n_failed": 0,
            "n_invalid_labels": 0,
            "observed_sampling_rates": [],
            "observed_lead_orders": [],
            "signal_shape_observed": None,
            "failed_records": [],
            "warnings": ["no_external_records_found"],
        }
        return updated, summary

    for idx, row in updated.iterrows():
        if row.get("status") != "ready":
            continue
        record_path = Path(str(row["record_path"]))
        cache_path = cache_root / str(row["source"]) / f"{_safe_cache_name(str(row['external_record_id']))}.npy"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            signal, metadata = standardize_external_signal(
                record_path,
                expected_lead_order=STANDARD_12_LEAD_ORDER,
                target_sampling_rate=int(external_cfg["sampling_rate"]),
                signal_length=int(external_cfg["signal_length"]),
                normalize=bool(config["data"].get("normalize_per_lead", True)),
            )
            np.save(cache_path, signal.astype(np.float32))
            updated.at[idx, "cache_path"] = str(cache_path)
            updated.at[idx, "status"] = "cached"
            observed_sampling_rates.add(float(metadata["sampling_rate_original"]))
            observed_shapes.append(list(metadata["observed_signal_shape"]))
            if metadata["lead_order_original"] not in observed_lead_orders:
                observed_lead_orders.append(list(metadata["lead_order_original"]))
            nan_count += int(np.isnan(signal).sum())
            inf_count += int(np.isinf(signal).sum())
        except Exception as exc:
            updated.at[idx, "status"] = "failed"
            updated.at[idx, "warnings"] = "|".join(
                part for part in [str(row.get("warnings", "")).strip("|"), "cache_failed"] if part
            )
            failed_records.append(
                {
                    "external_record_id": str(row["external_record_id"]),
                    "record_path": str(record_path),
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
            )

    if len(observed_lead_orders) > 1:
        warnings.append("inconsistent_lead_order_observed")
    if any(abs(rate - float(external_cfg["sampling_rate"])) > 1e-6 for rate in observed_sampling_rates):
        warnings.append("sampling_rate_resampling_applied")
    if any(shape != [int(external_cfg["num_leads"]), int(external_cfg["signal_length"])] for shape in observed_shapes):
        warnings.append("unexpected_cached_signal_shape")

    summary = {
        "enabled": bool(external_cfg.get("enabled", False)),
        "root": external_cfg.get("root"),
        "sources_requested": list(external_cfg.get("sources", [])),
        "sources_found": sorted(updated["source"].dropna().astype(str).unique().tolist()),
        "n_records_found": int(len(updated)),
        "n_cached": int((updated["status"] == "cached").sum()),
        "n_failed": int(len(failed_records)),
        "n_invalid_labels": int((updated["status"] == "invalid_label").sum()),
        "observed_sampling_rates": sorted(float(rate) for rate in observed_sampling_rates),
        "observed_lead_orders": observed_lead_orders,
        "signal_shape_observed": observed_shapes[0] if observed_shapes else None,
        "failed_records": failed_records,
        "warnings": warnings,
        "dtype": "float32" if observed_shapes else None,
        "nan_count": int(nan_count),
        "inf_count": int(inf_count),
    }
    return updated, summary
