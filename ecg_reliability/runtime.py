from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ecg_reliability.config import apply_overrides, load_config
from ecg_reliability.models.factory import build_torch_model
from ecg_reliability.paths import ensure_output_structure, resolve_path
from ecg_reliability.seed import seed_everything
from ecg_reliability.training.checkpoints import load_checkpoint


def load_runtime_config(config_path: str, overrides: list[str] | None = None) -> dict:
    config = load_config(config_path)
    if overrides:
        config = apply_overrides(config, overrides)
    ensure_output_structure()
    seed_everything(int(config["project"]["seed"]))
    return config


def model_name_from_config(config: dict, override: str | None = None) -> str:
    return override or config["model"]["name"]


def checkpoint_dir(model_name: str) -> Path:
    return resolve_path(f"outputs/checkpoints/{model_name}")


def prediction_dir(model_name: str, namespace: str = "raw") -> Path:
    return resolve_path(f"outputs/predictions/{model_name}/{namespace}")


def calibration_dir(model_name: str) -> Path:
    return resolve_path(f"outputs/calibration/{model_name}")


def decision_dir(model_name: str) -> Path:
    return resolve_path(f"outputs/decision/{model_name}")


def robustness_dir(model_name: str) -> Path:
    return resolve_path(f"outputs/robustness/{model_name}")


def explanations_dir(model_name: str) -> Path:
    return resolve_path(f"outputs/explanations/{model_name}")


def load_prediction_arrays(model_name: str, split: str, namespace: str = "raw") -> dict[str, np.ndarray]:
    base = prediction_dir(model_name, namespace) / split
    return {
        "ecg_ids": np.load(base / "ecg_ids.npy"),
        "logits": np.load(base / "logits.npy"),
        "probabilities": np.load(base / "probabilities.npy"),
        "labels": np.load(base / "labels.npy"),
    }


def load_torch_model(checkpoint_path: str | Path, device: str | torch.device = "cpu"):
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = build_torch_model(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, config
