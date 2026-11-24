from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ecg_reliability.data.dataset import CachedECGDataset
from ecg_reliability.io import write_table
from ecg_reliability.models.factory import build_torch_model


def test_cached_dataset_shapes(tmp_path) -> None:
    signal_a = np.random.randn(12, 1000).astype(np.float32)
    signal_b = np.random.randn(12, 1000).astype(np.float32)
    path_a = tmp_path / "a.npy"
    path_b = tmp_path / "b.npy"
    np.save(path_a, signal_a)
    np.save(path_b, signal_b)
    manifest = pd.DataFrame(
        [
            {"ecg_id": 1, "cache_path": str(path_a), "split": "train", "NORM": 1, "MI": 0, "STTC": 0, "CD": 0, "HYP": 0},
            {"ecg_id": 2, "cache_path": str(path_b), "split": "train", "NORM": 0, "MI": 1, "STTC": 0, "CD": 0, "HYP": 1},
        ]
    )
    manifest_path = tmp_path / "manifest.csv"
    write_table(manifest, manifest_path)
    dataset = CachedECGDataset(manifest_path, split="train")
    signal, labels, metadata = dataset[0]
    assert signal.shape == (12, 1000)
    assert labels.shape == (5,)
    assert metadata["ecg_id"] == 1


def test_all_torch_models_output_shape() -> None:
    batch = torch.randn(4, 12, 1000)
    for model_name in ["simple_cnn", "resnet1d", "inceptiontime", "tcn", "transformer"]:
        config = {
            "data": {"num_leads": 12, "target_labels": ["NORM", "MI", "STTC", "CD", "HYP"]},
            "model": {
                "name": model_name,
                "num_classes": 5,
                "hidden_dim": 64,
                "dropout": 0.1,
                "num_heads": 4,
                "num_layers": 2,
            },
        }
        model = build_torch_model(config)
        logits = model(batch)
        assert logits.shape == (4, 5)
