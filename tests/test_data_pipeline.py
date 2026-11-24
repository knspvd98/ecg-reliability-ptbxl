from __future__ import annotations

import numpy as np

from ecg_reliability.data.features import compute_engineered_features
from ecg_reliability.data.preprocessing import normalize_per_lead, pad_or_trim_signal


def test_normalize_per_lead_is_finite_for_constant_signal() -> None:
    signal = np.ones((12, 1000), dtype=np.float32)
    normalized = normalize_per_lead(signal)
    assert normalized.shape == signal.shape
    assert np.isfinite(normalized).all()
    assert np.allclose(normalized, 0.0)


def test_pad_or_trim_signal_preserves_requested_shape() -> None:
    short = np.ones((12, 900), dtype=np.float32)
    long = np.ones((12, 1100), dtype=np.float32)
    assert pad_or_trim_signal(short, 1000).shape == (12, 1000)
    assert pad_or_trim_signal(long, 1000).shape == (12, 1000)


def test_engineered_features_are_deterministic_and_finite() -> None:
    signal = np.random.default_rng(42).normal(size=(12, 1000)).astype(np.float32)
    first = compute_engineered_features(signal)
    second = compute_engineered_features(signal)
    assert first.shape == (12 * 13,)
    assert np.isfinite(first).all()
    assert np.array_equal(first, second)
