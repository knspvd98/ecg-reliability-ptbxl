from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import butter, filtfilt

LIMB_LEADS = [0, 1, 2, 3, 4, 5]
PRECORDIAL_LEADS = [6, 7, 8, 9, 10, 11]

# Empirical reference metadata for the PhysioNet MIT-BIH Noise Stress Test Database (Moody et al., 1984).
# Synthetic perturbations below are parameterized by SNR dB and spectral characteristics inspired by NSTDB;
# they are stress tests for ML reliability analysis and MUST NOT be interpreted as clinical equivalence claims.
MIT_BIH_NSTDB_REFERENCE: dict[str, Any] = {
    "database": "MIT-BIH Noise Stress Test Database (nstdb)",
    "physionet_url": "https://physionet.org/content/nstdb/1.0.0/",
    "native_sampling_rate_hz": 360,
    "noise_records": {
        "bw": "Baseline wander (low-frequency respiratory/postural drift, < 1 Hz)",
        "ma": "Muscle artifact (high-frequency skeletal EMG contamination, typically 20-150 Hz; band-limited at Nyquist)",
        "em": "Electrode motion artifact (transient high-amplitude skin-electrode impedance shifts mimicking ectopic beats)",
    },
    "canonical_snr_db_levels": [24.0, 18.0, 12.0, 6.0, 0.0, -6.0],
    "scientific_caveat": (
        "Synthetic perturbations approximate physical ECG artifacts in normalized SD or mV units. "
        "Degradation under synthetic stress tests characterizes model sensitivity and does not establish "
        "clinical robustness on real ambulatory recordings."
    ),
}


def _ensure_2d(signal: np.ndarray) -> np.ndarray:
    arr = np.asarray(signal, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D ECG signal (leads, samples), got shape {arr.shape}")
    return arr


def _scale_noise_to_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float, eps: float = 1e-8) -> np.ndarray:
    """Scale additive noise per lead to achieve target SNR in decibels: SNR_dB = 10 * log10(P_sig / P_noise)."""
    sig_power = np.mean(signal.astype(np.float64) ** 2, axis=1, keepdims=True)
    noise_power = np.mean(noise.astype(np.float64) ** 2, axis=1, keepdims=True)
    target_noise_power = sig_power / (10.0 ** (float(snr_db) / 10.0))
    scale = np.sqrt(np.where(noise_power > eps, target_noise_power / np.maximum(noise_power, eps), 0.0))
    return (noise.astype(np.float64) * scale).astype(signal.dtype)


def lead_dropout_single(signal: np.ndarray, lead_index: int) -> np.ndarray:
    """Zero out a single ECG lead (simulates disconnected electrode / open lead)."""
    perturbed = _ensure_2d(signal).copy()
    if not (0 <= int(lead_index) < perturbed.shape[0]):
        raise ValueError(f"lead_index {lead_index} out of range for {perturbed.shape[0]} leads")
    perturbed[int(lead_index), :] = 0.0
    return perturbed


def lead_dropout_group(signal: np.ndarray, group: str) -> np.ndarray:
    """Zero out an anatomical lead group: ``limb`` (leads 0-5) or ``precordial`` (leads 6-11)."""
    perturbed = _ensure_2d(signal).copy()
    if group == "limb":
        indices = [i for i in LIMB_LEADS if i < perturbed.shape[0]]
    elif group == "precordial":
        indices = [i for i in PRECORDIAL_LEADS if i < perturbed.shape[0]]
    else:
        raise ValueError(f"Unknown lead group: {group!r} (expected 'limb' or 'precordial')")
    perturbed[indices, :] = 0.0
    return perturbed


def gaussian_noise(
    signal: np.ndarray,
    std: float | None = None,
    *,
    snr_db: float | None = None,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Add broadband zero-mean Gaussian noise calibrated by ``std`` (in signal units) or ``snr_db``."""
    arr = _ensure_2d(signal)
    generator = rng if rng is not None else np.random.default_rng(seed)
    unit_noise = generator.normal(loc=0.0, scale=1.0, size=arr.shape).astype(arr.dtype)
    if snr_db is not None:
        scaled = _scale_noise_to_snr(arr, unit_noise, snr_db=float(snr_db))
        return (arr + scaled).astype(arr.dtype)
    if std is None:
        raise ValueError("gaussian_noise requires either std or snr_db.")
    return (arr + unit_noise * float(std)).astype(arr.dtype)


def baseline_wander(
    signal: np.ndarray,
    amplitude: float,
    frequency_hz: float,
    sampling_rate: int = 100,
    *,
    phase: float = 0.0,
    multi_harmonic: bool = False,
) -> np.ndarray:
    """Add low-frequency respiratory/movement baseline wander (units match input signal: SD or mV)."""
    perturbed = _ensure_2d(signal).copy()
    t = np.arange(perturbed.shape[1], dtype=np.float32) / float(sampling_rate)
    drift = float(amplitude) * np.sin(2.0 * np.pi * float(frequency_hz) * t + float(phase))
    if multi_harmonic:
        drift = drift + 0.35 * float(amplitude) * np.sin(2.0 * np.pi * (0.5 * float(frequency_hz)) * t + 0.7)
    return (perturbed + drift[None, :]).astype(perturbed.dtype)


def powerline_interference(
    signal: np.ndarray,
    amplitude: float,
    frequency_hz: float = 50.0,
    sampling_rate: int = 100,
    *,
    phase: float = 0.25,
) -> np.ndarray:
    """Add sinusoidal AC mains interference (50 Hz or 60 Hz).

    Physical note: at 100 Hz sampling rate (Nyquist = 50 Hz), a 50 Hz sinusoid sits at
    Nyquist (amplitude depends on phase; phase=0.25 rad avoids zero-crossing degeneracy)
    and 60 Hz aliases to 40 Hz. At 500 Hz sampling rate, both 50 Hz and 60 Hz are
    fully resolved below Nyquist.
    """
    perturbed = _ensure_2d(signal).copy()
    t = np.arange(perturbed.shape[1], dtype=np.float64) / float(sampling_rate)
    interference = float(amplitude) * np.sin(2.0 * np.pi * float(frequency_hz) * t + float(phase))
    return (perturbed + interference[None, :].astype(perturbed.dtype)).astype(perturbed.dtype)


def muscle_noise(
    signal: np.ndarray,
    std: float | None = None,
    *,
    snr_db: float | None = None,
    sampling_rate: int = 100,
    lowcut_hz: float = 20.0,
    highcut_hz: float = 45.0,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Add band-limited high-frequency skeletal muscle (EMG) noise (inspired by MIT-BIH NSTDB ``ma``)."""
    arr = _ensure_2d(signal)
    generator = rng if rng is not None else np.random.default_rng(seed)
    raw = generator.normal(loc=0.0, scale=1.0, size=arr.shape)
    nyquist = 0.5 * float(sampling_rate)
    lo = max(1.0, min(float(lowcut_hz), nyquist * 0.85))
    hi = min(float(highcut_hz), nyquist * 0.96)
    if hi > lo + 1.0 and arr.shape[1] > 18:
        b, a = butter(3, [lo / nyquist, hi / nyquist], btype="bandpass")
        filtered = filtfilt(b, a, raw, axis=1)
    else:
        filtered = raw
    # Normalize filtered noise to unit std per lead before scaling
    filt_std = np.std(filtered, axis=1, keepdims=True)
    filtered = np.where(filt_std > 1e-8, filtered / np.maximum(filt_std, 1e-8), 0.0).astype(arr.dtype)
    if snr_db is not None:
        return (arr + _scale_noise_to_snr(arr, filtered, snr_db=float(snr_db))).astype(arr.dtype)
    scale = float(std) if std is not None else 0.1
    return (arr + filtered * scale).astype(arr.dtype)


def electrode_motion(
    signal: np.ndarray,
    amplitude: float = 0.5,
    *,
    n_events: int = 2,
    event_duration_s: float = 0.6,
    sampling_rate: int = 100,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Add transient skin-electrode motion impulses (inspired by MIT-BIH NSTDB ``em``)."""
    arr = _ensure_2d(signal).copy()
    generator = rng if rng is not None else np.random.default_rng(seed)
    n_leads, n_samples = arr.shape
    width = max(4, min(n_samples // 2, int(round(float(event_duration_s) * float(sampling_rate)))))
    t_win = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    # Asymmetric biphasic transient envelope typical of electrode slip
    waveform = np.exp(-3.0 * t_win**2) * np.sin(np.pi * t_win)
    waveform = waveform / max(float(np.max(np.abs(waveform))), 1e-6)
    for _ in range(max(1, int(n_events))):
        start = int(generator.integers(0, max(1, n_samples - width)))
        affected_leads = generator.choice(n_leads, size=max(1, n_leads // 3), replace=False)
        sign = float(generator.choice([-1.0, 1.0]))
        arr[affected_leads, start : start + width] += sign * float(amplitude) * waveform[None, :]
    return arr.astype(signal.dtype)


def amplitude_scaling(signal: np.ndarray, scale: float) -> np.ndarray:
    """Multiply all leads by a uniform gain factor ``scale``."""
    return (_ensure_2d(signal) * float(scale)).astype(signal.dtype)


def build_severity_perturbation_suite(
    sampling_rate: int = 100,
    seed: int = 42,
) -> dict[str, dict[str, Any]]:
    """Return a structured multi-severity stress-test suite with documented units & severity tiers."""
    return {
        "baseline_wander_mild": {
            "type": "baseline_wander",
            "severity": "mild",
            "amplitude": 0.15,
            "frequency_hz": 0.25,
            "sampling_rate": sampling_rate,
            "units": "signal_units (SD or mV)",
        },
        "baseline_wander_moderate": {
            "type": "baseline_wander",
            "severity": "moderate",
            "amplitude": 0.40,
            "frequency_hz": 0.33,
            "sampling_rate": sampling_rate,
            "multi_harmonic": True,
            "units": "signal_units (SD or mV)",
        },
        "baseline_wander_severe": {
            "type": "baseline_wander",
            "severity": "severe",
            "amplitude": 0.85,
            "frequency_hz": 0.45,
            "sampling_rate": sampling_rate,
            "multi_harmonic": True,
            "units": "signal_units (SD or mV)",
        },
        "powerline_50hz_mild": {
            "type": "powerline_interference",
            "severity": "mild",
            "amplitude": 0.10,
            "frequency_hz": 50.0,
            "sampling_rate": sampling_rate,
            "units": "signal_units (SD or mV)",
        },
        "powerline_60hz_moderate": {
            "type": "powerline_interference",
            "severity": "moderate",
            "amplitude": 0.25,
            "frequency_hz": 60.0,
            "sampling_rate": sampling_rate,
            "units": "signal_units (SD or mV)",
        },
        "muscle_noise_18db": {
            "type": "muscle_noise",
            "severity": "mild",
            "snr_db": 18.0,
            "sampling_rate": sampling_rate,
            "seed": seed,
            "units": "SNR_dB",
        },
        "muscle_noise_12db": {
            "type": "muscle_noise",
            "severity": "moderate",
            "snr_db": 12.0,
            "sampling_rate": sampling_rate,
            "seed": seed,
            "units": "SNR_dB",
        },
        "muscle_noise_6db": {
            "type": "muscle_noise",
            "severity": "severe",
            "snr_db": 6.0,
            "sampling_rate": sampling_rate,
            "seed": seed,
            "units": "SNR_dB",
        },
        "electrode_motion_moderate": {
            "type": "electrode_motion",
            "severity": "moderate",
            "amplitude": 0.75,
            "n_events": 3,
            "sampling_rate": sampling_rate,
            "seed": seed,
            "units": "signal_units (SD or mV)",
        },
        "lead_dropout_single_v1": {
            "type": "lead_dropout_single",
            "severity": "moderate",
            "lead_index": 6,
            "units": "binary_lead_mask",
        },
        "lead_dropout_group_limb": {
            "type": "lead_dropout_group",
            "severity": "severe",
            "group": "limb",
            "units": "binary_lead_mask",
        },
        "lead_dropout_group_precordial": {
            "type": "lead_dropout_group",
            "severity": "severe",
            "group": "precordial",
            "units": "binary_lead_mask",
        },
        "amplitude_scaling_attenuated": {
            "type": "amplitude_scaling",
            "severity": "mild",
            "scale": 0.75,
            "units": "dimensionless_gain",
        },
        "amplitude_scaling_amplified": {
            "type": "amplitude_scaling",
            "severity": "mild",
            "scale": 1.25,
            "units": "dimensionless_gain",
        },
    }


def load_nstdb_noise_reference(nstdb_root: str | Path | None) -> dict[str, Any]:
    """Inspect a local MIT-BIH Noise Stress Test Database directory if supplied by the researcher.

    Never downloads data automatically; reports availability and record metadata only.
    """
    if not nstdb_root:
        return {
            "available": False,
            "reason": "NSTDB_ROOT not provided; using synthetic stress-test parameterizations.",
            "reference": MIT_BIH_NSTDB_REFERENCE,
        }
    root = Path(nstdb_root)
    if not root.exists():
        return {
            "available": False,
            "reason": f"Configured NSTDB_ROOT does not exist: {root}",
            "reference": MIT_BIH_NSTDB_REFERENCE,
        }
    found = [rec for rec in ("bw", "em", "ma") if (root / f"{rec}.hea").exists()]
    return {
        "available": len(found) > 0,
        "root": str(root),
        "noise_records_found": found,
        "reference": MIT_BIH_NSTDB_REFERENCE,
    }


def apply_perturbation(signal: np.ndarray, name: str, **kwargs) -> np.ndarray:
    """Apply a named ECG perturbation to a single ``(leads, samples)`` waveform."""
    kind = str(kwargs.get("type") or name)
    if kind.startswith("lead_dropout_group_"):
        kind = "lead_dropout_group"
    elif kind.startswith("lead_dropout_single"):
        kind = "lead_dropout_single"
    elif kind.startswith("baseline_wander"):
        kind = "baseline_wander"
    elif kind.startswith("powerline"):
        kind = "powerline_interference"
    elif kind.startswith("muscle_noise"):
        kind = "muscle_noise"
    elif kind.startswith("electrode_motion"):
        kind = "electrode_motion"
    elif kind.startswith("amplitude_scaling"):
        kind = "amplitude_scaling"
    elif kind.startswith("gaussian_noise"):
        kind = "gaussian_noise"

    if kind == "lead_dropout_single":
        return lead_dropout_single(signal, lead_index=int(kwargs.get("lead_index", 0)))
    if kind == "lead_dropout_group":
        return lead_dropout_group(signal, group=str(kwargs.get("group", "limb")))
    if kind == "gaussian_noise":
        return gaussian_noise(
            signal,
            std=float(kwargs["std"]) if kwargs.get("std") is not None else None,
            snr_db=float(kwargs["snr_db"]) if kwargs.get("snr_db") is not None else None,
            rng=kwargs.get("rng"),
            seed=kwargs.get("seed"),
        )
    if kind == "baseline_wander":
        return baseline_wander(
            signal,
            amplitude=float(kwargs["amplitude"]),
            frequency_hz=float(kwargs["frequency_hz"]),
            sampling_rate=int(kwargs.get("sampling_rate", 100)),
            phase=float(kwargs.get("phase", 0.0)),
            multi_harmonic=bool(kwargs.get("multi_harmonic", False)),
        )
    if kind == "powerline_interference":
        return powerline_interference(
            signal,
            amplitude=float(kwargs["amplitude"]),
            frequency_hz=float(kwargs.get("frequency_hz", 50.0)),
            sampling_rate=int(kwargs.get("sampling_rate", 100)),
            phase=float(kwargs.get("phase", 0.25)),
        )
    if kind == "muscle_noise":
        return muscle_noise(
            signal,
            std=float(kwargs["std"]) if kwargs.get("std") is not None else None,
            snr_db=float(kwargs["snr_db"]) if kwargs.get("snr_db") is not None else None,
            sampling_rate=int(kwargs.get("sampling_rate", 100)),
            lowcut_hz=float(kwargs.get("lowcut_hz", 20.0)),
            highcut_hz=float(kwargs.get("highcut_hz", 45.0)),
            rng=kwargs.get("rng"),
            seed=kwargs.get("seed"),
        )
    if kind == "electrode_motion":
        return electrode_motion(
            signal,
            amplitude=float(kwargs.get("amplitude", 0.5)),
            n_events=int(kwargs.get("n_events", 2)),
            event_duration_s=float(kwargs.get("event_duration_s", 0.6)),
            sampling_rate=int(kwargs.get("sampling_rate", 100)),
            rng=kwargs.get("rng"),
            seed=kwargs.get("seed"),
        )
    if kind == "amplitude_scaling":
        return amplitude_scaling(signal, scale=float(kwargs["scale"]))
    raise ValueError(f"Unknown perturbation: {name}")
