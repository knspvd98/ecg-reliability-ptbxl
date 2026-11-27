from __future__ import annotations

from collections.abc import Sequence
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import resample, resample_poly

STANDARD_12_LEAD_ORDER = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
DISPLAY_12_LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

_LEAD_ALIASES = {
    "AVR": "AVR",
    "AVL": "AVL",
    "AVF": "AVF",
    "A VR": "AVR",
    "A VL": "AVL",
    "A VF": "AVF",
    "LEAD I": "I",
    "LEAD II": "II",
    "LEAD III": "III",
}

_UNIT_SCALE_TO_MV = {
    "mv": 1.0,
    "uv": 1e-3,
    "µv": 1e-3,
    "v": 1e3,
}


def _require_wfdb():
    try:
        import wfdb
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("wfdb is required for reading WFDB ECG records.") from exc
    return wfdb


def canonical_lead_name(name: str) -> str:
    cleaned = str(name).strip().upper()
    return _LEAD_ALIASES.get(cleaned, cleaned)


def inspect_wfdb_header(record_path: str | Path) -> dict[str, Any]:
    """Read WFDB header metadata without loading waveform samples."""
    wfdb = _require_wfdb()
    header = wfdb.rdheader(str(record_path))
    raw_lead_order = [str(name) for name in list(getattr(header, "sig_name", []) or [])]
    canonical_leads = [canonical_lead_name(name) for name in raw_lead_order]
    raw_units = [str(unit) for unit in list(getattr(header, "units", []) or [])]
    adc_gain = [float(val) for val in list(getattr(header, "adc_gain", []) or [])]
    baseline = [float(val) for val in list(getattr(header, "baseline", []) or [])]
    fs_value = getattr(header, "fs", None)
    sig_len = getattr(header, "sig_len", None)
    n_sig = getattr(header, "n_sig", None)
    return {
        "sampling_rate": float(fs_value) if fs_value is not None else None,
        "signal_length": int(sig_len) if sig_len is not None else None,
        "num_leads": int(n_sig) if n_sig is not None else len(raw_lead_order),
        "lead_order": raw_lead_order,
        "canonical_lead_order": canonical_leads,
        "lead_order_matches_standard": canonical_leads == STANDARD_12_LEAD_ORDER,
        "units": raw_units,
        "adc_gain": adc_gain,
        "baseline": baseline,
    }


def pad_or_trim_signal(signal: np.ndarray, target_length: int) -> np.ndarray:
    """Right-pad with zeros or truncate along the time axis to ``target_length``."""
    arr = np.asarray(signal, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D signal (leads, samples), got shape {arr.shape}")
    if target_length <= 0:
        raise ValueError(f"target_length must be positive, got {target_length}")
    current_length = arr.shape[1]
    if current_length == target_length:
        return arr.copy()
    if current_length > target_length:
        return arr[:, :target_length].copy()
    pad_width = target_length - current_length
    return np.pad(arr, ((0, 0), (0, pad_width)), mode="constant", constant_values=0.0).astype(np.float32)


def normalize_per_lead(signal: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Zero-mean, unit-variance per-lead z-normalization with safe constant-lead handling.

    Note: after z-normalization, amplitudes are expressed in dimensionless per-lead
    standard deviations (SD) rather than raw physical millivolts (mV).
    """
    arr = np.asarray(signal, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D signal (leads, samples), got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("Input signal contains NaN or Inf values before normalization.")
    mean = arr.mean(axis=1, keepdims=True)
    std = arr.std(axis=1, keepdims=True)
    centered = arr - mean
    safe_std = np.where(std > eps, std, 1.0)
    normalized = np.where(std > eps, centered / safe_std, 0.0)
    return normalized.astype(np.float32)


def resample_signal(signal: np.ndarray, original_fs: float, target_fs: float) -> np.ndarray:
    """Resample a 2D ``(leads, samples)`` ECG signal using polyphase filtering when integer-ratio."""
    arr = np.asarray(signal, dtype=np.float32)
    if original_fs <= 0 or target_fs <= 0:
        raise ValueError(f"Sampling rates must be positive: original_fs={original_fs}, target_fs={target_fs}")
    if abs(float(original_fs) - float(target_fs)) < 1e-6:
        return arr.copy()
    orig_int = int(round(original_fs))
    targ_int = int(round(target_fs))
    if abs(original_fs - orig_int) < 1e-6 and abs(target_fs - targ_int) < 1e-6:
        divisor = gcd(orig_int, targ_int)
        up = targ_int // divisor
        down = orig_int // divisor
        return resample_poly(arr, up=up, down=down, axis=1).astype(np.float32)
    target_len = max(1, int(round(arr.shape[1] * float(target_fs) / float(original_fs))))
    return resample(arr, target_len, axis=1).astype(np.float32)


def reorder_leads(
    signal: np.ndarray,
    observed_lead_names: Sequence[str],
    expected_lead_order: Sequence[str] = STANDARD_12_LEAD_ORDER,
) -> np.ndarray:
    """Reorder a ``(leads, samples)`` array to match ``expected_lead_order`` canonically."""
    arr = np.asarray(signal, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D signal (leads, samples), got shape {arr.shape}")
    if arr.shape[0] != len(observed_lead_names):
        raise ValueError(
            f"Signal lead count ({arr.shape[0]}) does not match observed_lead_names ({len(observed_lead_names)})."
        )
    canonical_observed = [canonical_lead_name(name) for name in observed_lead_names]
    if len(set(canonical_observed)) != len(canonical_observed):
        raise ValueError(f"Duplicate lead names observed: {observed_lead_names}")
    index_map = {name: idx for idx, name in enumerate(canonical_observed)}
    canonical_expected = [canonical_lead_name(name) for name in expected_lead_order]
    missing = [lead for lead in canonical_expected if lead not in index_map]
    if missing:
        raise ValueError(f"Missing required ECG leads: {missing} (observed={canonical_observed})")
    order = [index_map[lead] for lead in canonical_expected]
    return arr[order, :].copy()


def _convert_units_to_mv(signal: np.ndarray, units: Sequence[str] | None) -> tuple[np.ndarray, list[str]]:
    if not units:
        return signal, []
    normalized_units = [str(u).strip() for u in units]
    if len(normalized_units) != signal.shape[0]:
        return signal, normalized_units
    converted = signal.copy()
    for idx, unit in enumerate(normalized_units):
        key = unit.lower()
        if key not in _UNIT_SCALE_TO_MV:
            raise ValueError(f"Unsupported physical unit for lead {idx}: {unit!r} (expected mV, uV, or V)")
        scale = _UNIT_SCALE_TO_MV[key]
        if scale != 1.0:
            converted[idx] = converted[idx] * scale
    return converted, ["mV"] * signal.shape[0]


def load_ptbxl_signal(
    record_path: str | Path,
    *,
    signal_length: int = 1000,
    normalize: bool = True,
    expected_sampling_rate: float | None = None,
    expected_lead_order: Sequence[str] = STANDARD_12_LEAD_ORDER,
    enforce_lead_order: bool = True,
    convert_to_mv: bool = True,
) -> np.ndarray:
    """Load and standardize a PTB-XL WFDB signal to ``(12, signal_length)`` float32.

    Physical units are converted to millivolts (mV) before optional per-lead z-normalization.
    Lead order is canonically validated and reordered to standard 12-lead order when header
    lead names are available.
    """
    wfdb = _require_wfdb()
    samples, fields = wfdb.rdsamp(str(record_path))
    signal = np.asarray(samples, dtype=np.float32).T
    if signal.ndim != 2:
        raise ValueError(f"Unexpected WFDB signal dimensions for {record_path}: {signal.shape}")
    if not np.isfinite(signal).all():
        raise ValueError(f"WFDB record contains non-finite values: {record_path}")

    observed_leads = [str(name) for name in list(fields.get("sig_name", []) or [])]
    observed_units = [str(unit) for unit in list(fields.get("units", []) or [])]
    if convert_to_mv and observed_units:
        signal, _ = _convert_units_to_mv(signal, observed_units)

    if enforce_lead_order and observed_leads:
        signal = reorder_leads(signal, observed_leads, expected_lead_order=expected_lead_order)
    elif signal.shape[0] != len(expected_lead_order):
        raise ValueError(f"Expected {len(expected_lead_order)} leads, got {signal.shape[0]} for {record_path}")

    observed_fs = fields.get("fs")
    if expected_sampling_rate is not None and observed_fs is not None:
        signal = resample_signal(signal, float(observed_fs), float(expected_sampling_rate))

    signal = pad_or_trim_signal(signal, target_length=int(signal_length))
    if normalize:
        signal = normalize_per_lead(signal)
    return signal.astype(np.float32)
