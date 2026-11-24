from __future__ import annotations

import numpy as np
import torch
from torch import nn


def compute_pos_weight(targets: np.ndarray, eps: float = 1e-6) -> torch.Tensor:
    positives = targets.sum(axis=0)
    negatives = targets.shape[0] - positives
    weights = negatives / np.maximum(positives, eps)
    return torch.tensor(weights, dtype=torch.float32)


def make_bce_loss(pos_weight: torch.Tensor | None = None) -> nn.Module:
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
