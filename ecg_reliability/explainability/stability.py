from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    denom = (np.linalg.norm(a_flat) * np.linalg.norm(b_flat)) + eps
    return float(np.dot(a_flat, b_flat) / denom)


def spearman_similarity(a: np.ndarray, b: np.ndarray) -> float:
    correlation = spearmanr(a.reshape(-1), b.reshape(-1)).correlation
    return float(correlation if correlation is not None else np.nan)


def topk_overlap(a: np.ndarray, b: np.ndarray, k: int = 100) -> float:
    a_idx = np.argsort(np.abs(a.reshape(-1)))[-k:]
    b_idx = np.argsort(np.abs(b.reshape(-1)))[-k:]
    return float(len(set(a_idx.tolist()) & set(b_idx.tolist())) / max(k, 1))


def lead_attribution_entropy(attribution: np.ndarray, eps: float = 1e-8) -> float:
    per_lead = np.abs(attribution).sum(axis=1)
    probs = per_lead / max(per_lead.sum(), eps)
    return float(-(probs * np.log(probs + eps)).sum())
