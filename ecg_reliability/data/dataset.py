from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ecg_reliability.data.labels import TARGET_LABELS
from ecg_reliability.io import read_table


class CachedECGDataset(Dataset[tuple[torch.Tensor, torch.Tensor, dict[str, Any]]]):
    """PyTorch Dataset backed by cached ``(12, T)`` float32 NumPy signal arrays."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: str | None = None,
        limit: int | None = None,
        label_names: list[str] | None = None,
        expected_num_leads: int | None = 12,
        validate_finite: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.label_names = list(label_names or TARGET_LABELS)
        self.expected_num_leads = expected_num_leads
        self.validate_finite = validate_finite

        frame = read_table(self.manifest_path)
        required_cols = {"ecg_id", "cache_path", *self.label_names}
        missing = sorted(required_cols - set(frame.columns))
        if missing:
            raise ValueError(f"Cache manifest {self.manifest_path} is missing required columns: {missing}")

        if split is not None:
            if "split" not in frame.columns:
                raise ValueError(f"Cache manifest {self.manifest_path} does not contain a 'split' column.")
            frame = frame.loc[frame["split"] == split]
        if limit is not None:
            frame = frame.iloc[: int(limit)]
        self.frame = frame.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        row = self.frame.iloc[index]
        cache_path = Path(str(row["cache_path"]))
        if not cache_path.exists():
            raise FileNotFoundError(f"Missing cached ECG signal file: {cache_path}")
        signal = np.load(cache_path).astype(np.float32, copy=False)
        if signal.ndim != 2:
            raise ValueError(f"Expected 2D cached signal (leads, samples) at {cache_path}, got {signal.shape}")
        if self.expected_num_leads is not None and signal.shape[0] != self.expected_num_leads:
            raise ValueError(
                f"Unexpected lead count in {cache_path}: expected {self.expected_num_leads}, got {signal.shape[0]}"
            )
        if self.validate_finite and not np.isfinite(signal).all():
            raise ValueError(f"Cached signal contains NaN or Inf values: {cache_path}")

        labels = np.asarray([float(row[label]) for label in self.label_names], dtype=np.float32)
        metadata: dict[str, Any] = {
            "ecg_id": int(row["ecg_id"]),
            "split": str(row["split"]) if "split" in row.index else "unknown",
            "cache_path": str(cache_path),
        }
        if "patient_id" in row.index and not np.isnan(float(row["patient_id"])):
            metadata["patient_id"] = int(row["patient_id"])
        return torch.from_numpy(np.ascontiguousarray(signal)), torch.from_numpy(labels), metadata


def manifest_targets(
    manifest_path: str | Path,
    split: str | None = None,
    label_names: list[str] | None = None,
) -> np.ndarray:
    """Return binary target matrix ``(N, K)`` from a cache manifest."""
    labels = list(label_names or TARGET_LABELS)
    frame = read_table(manifest_path)
    if split is not None and "split" in frame.columns:
        frame = frame.loc[frame["split"] == split]
    missing = [label for label in labels if label not in frame.columns]
    if missing:
        raise ValueError(f"Manifest {manifest_path} is missing label columns: {missing}")
    return frame[labels].to_numpy(dtype=np.float32)
