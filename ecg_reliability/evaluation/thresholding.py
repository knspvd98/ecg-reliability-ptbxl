from __future__ import annotations

import numpy as np


def sigmoid(logits: np.ndarray) -> np.ndarray:
    clipped = np.clip(logits, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def binarize(probabilities: np.ndarray, threshold: float | np.ndarray = 0.5) -> np.ndarray:
    return (probabilities >= threshold).astype(int)
