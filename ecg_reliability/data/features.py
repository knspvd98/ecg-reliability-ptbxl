from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.signal import welch

from ecg_reliability.data.preprocessing import DISPLAY_12_LEAD_NAMES

FEATURE_NAMES_PER_LEAD = [
    "mean",
    "std",
    "min",
    "max",
    "median",
    "q25",
    "q75",
    "iqr",
    "rms",
    "mean_abs",
    "zero_crossing_rate",
    "skewness",
    "kurtosis",
]


def _safe_higher_moments(lead: np.ndarray, eps: float = 1e-8) -> tuple[float, float]:
    centered = lead - float(np.mean(lead))
    std = float(np.std(centered))
    if std <= eps:
        return 0.0, 0.0
    z = centered / std
    skewness = float(np.mean(z**3))
    kurtosis = float(np.mean(z**4) - 3.0)
    return skewness, kurtosis


def compute_engineered_features(signal: np.ndarray) -> np.ndarray:
    """Extract 13 deterministic, finite statistical/morphological features per lead.

    For a 12-lead ECG, produces a ``(156,)`` float32 feature vector suitable for
    tabular baselines such as multi-label XGBoost.
    """
    arr = np.asarray(signal, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D signal (leads, samples), got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("Cannot compute engineered features on signal with NaN or Inf values.")

    features: list[float] = []
    for lead_idx in range(arr.shape[0]):
        lead = arr[lead_idx].astype(np.float64, copy=False)
        mean_val = float(np.mean(lead))
        std_val = float(np.std(lead))
        min_val = float(np.min(lead))
        max_val = float(np.max(lead))
        q25, median_val, q75 = (float(v) for v in np.percentile(lead, [25.0, 50.0, 75.0]))
        iqr_val = q75 - q25
        rms_val = float(np.sqrt(np.mean(lead * lead)))
        mean_abs_val = float(np.mean(np.abs(lead)))
        centered = lead - mean_val
        signs = np.signbit(centered)
        zcr_val = float(np.mean(signs[:-1] != signs[1:])) if lead.size > 1 else 0.0
        skew_val, kurt_val = _safe_higher_moments(lead)
        features.extend(
            [
                mean_val,
                std_val,
                min_val,
                max_val,
                median_val,
                q25,
                q75,
                iqr_val,
                rms_val,
                mean_abs_val,
                zcr_val,
                skew_val,
                kurt_val,
            ]
        )
    out = np.asarray(features, dtype=np.float32)
    if not np.isfinite(out).all():
        raise RuntimeError("Computed engineered features contain non-finite values.")
    return out


def compute_signal_quality_indicators(
    signal: np.ndarray,
    sampling_rate: int = 100,
    lead_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compute per-lead and aggregate ECG Signal Quality Indicators (SQIs).

    Inspired by Zhao & Zhang (2018) and NeuroKit2 ``ecg_quality`` heuristics:
    - ``kSQI``: excess kurtosis (sharp QRS morphology produces higher kurtosis than noise/flatline)
    - ``sSQI``: skewness of amplitude distribution
    - ``basSQI``: relative power retained outside the baseline-wander band (< 1 Hz), i.e. ``1 - P(<1 Hz) / P(0..40 Hz)``
    - ``pSQI``: QRS-band spectral concentration ``P(5..15 Hz) / P(5..40 Hz)``
    - ``hf_noise_ratio``: high-frequency EMG band ratio ``P(25..nyquist) / P(total)``
    - ``flatline``: lead standard deviation below 1e-4
    """
    arr = np.asarray(signal, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D signal (leads, samples), got shape {arr.shape}")
    names = list(lead_names) if lead_names is not None else DISPLAY_12_LEAD_NAMES[: arr.shape[0]]
    if len(names) < arr.shape[0]:
        names = [f"Lead {i + 1}" for i in range(arr.shape[0])]

    per_lead: list[dict[str, Any]] = []
    nperseg = min(256, arr.shape[1])
    nyquist = 0.5 * float(sampling_rate)

    for idx in range(arr.shape[0]):
        lead = arr[idx]
        non_finite = int((~np.isfinite(lead)).sum())
        clean = np.nan_to_num(lead, nan=0.0, posinf=0.0, neginf=0.0)
        std_val = float(np.std(clean))
        flatline = bool(std_val < 1e-4)
        rms_val = float(np.sqrt(np.mean(clean**2)))
        ptp_val = float(np.ptp(clean))
        s_sqi, k_sqi = _safe_higher_moments(clean)

        if flatline or arr.shape[1] < 8:
            bas_sqi = 0.0
            p_sqi = 0.0
            hf_ratio = 0.0
            pl_ratio = 0.0
        else:
            freqs, psd = welch(clean, fs=float(sampling_rate), nperseg=nperseg)
            total_mask = (freqs >= 0.0) & (freqs <= min(40.0, nyquist))
            total_power = float(np.trapezoid(psd[total_mask], freqs[total_mask])) if np.any(total_mask) else 0.0
            full_power = float(np.trapezoid(psd, freqs))
            bw_mask = freqs < 1.0
            bw_power = float(np.trapezoid(psd[bw_mask], freqs[bw_mask])) if np.any(bw_mask) else 0.0
            qrs_mask = (freqs >= 5.0) & (freqs <= min(15.0, nyquist))
            qrs_denom_mask = (freqs >= 5.0) & (freqs <= min(40.0, nyquist))
            qrs_power = float(np.trapezoid(psd[qrs_mask], freqs[qrs_mask])) if np.any(qrs_mask) else 0.0
            qrs_denom = float(np.trapezoid(psd[qrs_denom_mask], freqs[qrs_denom_mask])) if np.any(qrs_denom_mask) else 0.0
            hf_mask = freqs >= min(25.0, nyquist * 0.6)
            hf_power = float(np.trapezoid(psd[hf_mask], freqs[hf_mask])) if np.any(hf_mask) else 0.0
            # Check power near 50/60 Hz or its aliased frequency when fs=100 Hz
            pl_mask = (freqs >= max(nyquist - 2.0, 0.0)) | ((freqs >= 38.0) & (freqs <= 42.0))
            pl_power = float(np.trapezoid(psd[pl_mask], freqs[pl_mask])) if np.any(pl_mask) else 0.0

            bas_sqi = float(np.clip(1.0 - bw_power / max(total_power, 1e-12), 0.0, 1.0))
            p_sqi = float(np.clip(qrs_power / max(qrs_denom, 1e-12), 0.0, 1.0))
            hf_ratio = float(np.clip(hf_power / max(full_power, 1e-12), 0.0, 1.0))
            pl_ratio = float(np.clip(pl_power / max(full_power, 1e-12), 0.0, 1.0))

        # Heuristic quality tier (research indicator only, not a clinical device check)
        if non_finite > 0 or flatline:
            grade = "unacceptable"
        elif bas_sqi >= 0.70 and hf_ratio <= 0.35 and (k_sqi >= 1.0 or p_sqi >= 0.35):
            grade = "acceptable"
        else:
            grade = "borderline"

        per_lead.append(
            {
                "lead": names[idx],
                "lead_index": idx,
                "std": std_val,
                "rms": rms_val,
                "peak_to_peak": ptp_val,
                "kSQI": k_sqi,
                "sSQI": s_sqi,
                "basSQI": bas_sqi,
                "pSQI": p_sqi,
                "hf_noise_ratio": hf_ratio,
                "powerline_band_ratio": pl_ratio,
                "flatline": flatline,
                "non_finite_count": non_finite,
                "quality_grade": grade,
            }
        )

    n_flatline = sum(1 for item in per_lead if item["flatline"])
    n_unacceptable = sum(1 for item in per_lead if item["quality_grade"] == "unacceptable")
    n_acceptable = sum(1 for item in per_lead if item["quality_grade"] == "acceptable")
    return {
        "sampling_rate_hz": int(sampling_rate),
        "num_leads": int(arr.shape[0]),
        "num_samples": int(arr.shape[1]),
        "duration_seconds": float(arr.shape[1] / float(sampling_rate)),
        "mean_kSQI": float(np.mean([item["kSQI"] for item in per_lead])),
        "mean_basSQI": float(np.mean([item["basSQI"] for item in per_lead])),
        "mean_pSQI": float(np.mean([item["pSQI"] for item in per_lead])),
        "mean_hf_noise_ratio": float(np.mean([item["hf_noise_ratio"] for item in per_lead])),
        "flatline_leads": [item["lead"] for item in per_lead if item["flatline"]],
        "n_flatline_leads": int(n_flatline),
        "n_acceptable_leads": int(n_acceptable),
        "n_unacceptable_leads": int(n_unacceptable),
        "per_lead": per_lead,
    }
