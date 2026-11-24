from __future__ import annotations

import numpy as np

LIMB_LEADS = [0, 1, 2, 3, 4, 5]
PRECORDIAL_LEADS = [6, 7, 8, 9, 10, 11]


def lead_dropout_single(signal: np.ndarray, lead_index: int) -> np.ndarray:
    perturbed = signal.copy()
    perturbed[lead_index, :] = 0.0
    return perturbed


def lead_dropout_group(signal: np.ndarray, group: str) -> np.ndarray:
    perturbed = signal.copy()
    indices = LIMB_LEADS if group == "limb" else PRECORDIAL_LEADS
    perturbed[indices, :] = 0.0
    return perturbed


def gaussian_noise(signal: np.ndarray, std: float, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    return signal + rng.normal(loc=0.0, scale=std, size=signal.shape).astype(signal.dtype)


def baseline_wander(signal: np.ndarray, amplitude: float, frequency_hz: float, sampling_rate: int = 100) -> np.ndarray:
    perturbed = signal.copy()
    t = np.arange(signal.shape[1], dtype=np.float32) / float(sampling_rate)
    drift = amplitude * np.sin(2.0 * np.pi * frequency_hz * t)
    return perturbed + drift[None, :]


def amplitude_scaling(signal: np.ndarray, scale: float) -> np.ndarray:
    return signal * scale


def apply_perturbation(signal: np.ndarray, name: str, **kwargs) -> np.ndarray:
    if name.startswith("lead_dropout_group_"):
        name = "lead_dropout_group"
    if name == "lead_dropout_single":
        return lead_dropout_single(signal, lead_index=int(kwargs.get("lead_index", 0)))
    if name == "lead_dropout_group":
        return lead_dropout_group(signal, group=str(kwargs.get("group", "limb")))
    if name == "gaussian_noise":
        return gaussian_noise(signal, std=float(kwargs["std"]))
    if name == "baseline_wander":
        return baseline_wander(
            signal,
            amplitude=float(kwargs["amplitude"]),
            frequency_hz=float(kwargs["frequency_hz"]),
            sampling_rate=int(kwargs.get("sampling_rate", 100)),
        )
    if name == "amplitude_scaling":
        return amplitude_scaling(signal, scale=float(kwargs["scale"]))
    raise ValueError(f"Unknown perturbation: {name}")
