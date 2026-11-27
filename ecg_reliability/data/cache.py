from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ecg_reliability.data.labels import TARGET_LABELS
from ecg_reliability.data.preprocessing import STANDARD_12_LEAD_ORDER

CACHE_SCHEMA_VERSION = "2.0"


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def compute_preprocessing_fingerprint(config: dict[str, Any]) -> str:
    """Compute a deterministic SHA-256 fingerprint of dataset & preprocessing settings."""
    data_cfg = config.get("data", {})
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "dataset_identity": str(data_cfg.get("dataset_identity", "ptb-xl-1.0.3")),
        "sampling_rate": int(data_cfg.get("sampling_rate", 100)),
        "signal_length": int(data_cfg.get("signal_length", 1000)),
        "num_leads": int(data_cfg.get("num_leads", 12)),
        "lead_order": list(data_cfg.get("lead_order", STANDARD_12_LEAD_ORDER)),
        "normalize_per_lead": bool(data_cfg.get("normalize_per_lead", True)),
        "convert_to_mv": bool(data_cfg.get("convert_to_mv", True)),
        "enforce_lead_order": bool(data_cfg.get("enforce_lead_order", True)),
        "target_labels": list(data_cfg.get("target_labels", TARGET_LABELS)),
        "filter_empty_targets": bool(data_cfg.get("filter_empty_targets", True)),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def source_record_fingerprint(record_path: str | Path) -> dict[str, Any]:
    """Capture lightweight source file metadata (size & mtime_ns) for WFDB ``.hea``/``.dat`` pairs."""
    base = Path(record_path)
    files: dict[str, dict[str, int]] = {}
    for suffix in (".hea", ".dat"):
        candidate = base.with_suffix(suffix)
        if candidate.exists():
            stat = candidate.stat()
            files[suffix] = {
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
    return {
        "record_path": str(base),
        "files": files,
    }


def atomic_save_npy(path: str | Path, array: np.ndarray) -> None:
    """Atomically write a NumPy ``.npy`` file via same-directory temp file + fsync + replace."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.stem}.", suffix=".tmp", dir=str(target.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, np.ascontiguousarray(array))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Atomically write a JSON file via same-directory temp file + fsync + replace."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.stem}.", suffix=".tmp", dir=str(target.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def atomic_write_table(df: pd.DataFrame, path: str | Path) -> None:
    """Atomically write a CSV or Parquet table."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.stem}.", suffix=target.suffix, dir=str(target.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        if target.suffix == ".csv":
            df.to_csv(tmp_path, index=False)
        elif target.suffix == ".parquet":
            df.to_parquet(tmp_path, index=False)
        else:
            raise ValueError(f"Unsupported table extension: {target.suffix}")
        os.replace(tmp_path, target)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def load_cache_provenance(path: str | Path) -> dict[str, Any] | None:
    source = Path(path)
    if not source.exists():
        return None
    try:
        with source.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def validate_cached_signal(
    *,
    cache_path: str | Path,
    expected_fingerprint: str,
    expected_shape: tuple[int, int],
    expected_lead_order: list[str],
    expected_sampling_rate: float,
    expected_normalize: bool,
    expected_source_fingerprint: dict[str, Any],
    provenance_entry: dict[str, Any] | None,
) -> tuple[bool, str]:
    """Validate whether an existing cached signal can be safely reused.

    Refuses existence-only reuse: requires matching schema version, preprocessing
    fingerprint, sampling rate, normalization flag, canonical lead order, source
    record fingerprint, and a loadable finite float32 array of the expected shape.
    """
    target = Path(cache_path)
    if not target.exists():
        return False, "missing_cache_file"
    if provenance_entry is None:
        return False, "missing_provenance_entry"
    if provenance_entry.get("schema_version") != CACHE_SCHEMA_VERSION:
        return False, "schema_version_mismatch"
    if provenance_entry.get("preprocessing_fingerprint") != expected_fingerprint:
        return False, "preprocessing_fingerprint_mismatch"
    if bool(provenance_entry.get("normalize_per_lead")) != bool(expected_normalize):
        return False, "normalization_mismatch"
    if abs(float(provenance_entry.get("target_sampling_rate", -1.0)) - float(expected_sampling_rate)) > 1e-6:
        return False, "sampling_rate_mismatch"
    if list(provenance_entry.get("canonical_lead_order", [])) != list(expected_lead_order):
        return False, "lead_order_mismatch"
    if provenance_entry.get("source_fingerprint") != expected_source_fingerprint:
        return False, "source_record_changed"
    try:
        arr = np.load(target)
    except Exception:
        return False, "corrupted_npy_file"
    if tuple(int(d) for d in arr.shape) != tuple(int(d) for d in expected_shape):
        return False, "shape_mismatch"
    if str(arr.dtype) != "float32":
        return False, "dtype_mismatch"
    if not np.isfinite(arr).all():
        return False, "non_finite_values"
    return True, "valid"
