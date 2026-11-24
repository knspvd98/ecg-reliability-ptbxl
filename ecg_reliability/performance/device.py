from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def configure_torch_runtime(config: dict[str, Any]) -> dict[str, Any]:
    training_cfg = config["training"]
    deterministic = bool(training_cfg.get("deterministic", False))
    cudnn_benchmark = bool(training_cfg.get("cudnn_benchmark", not deterministic))
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = cudnn_benchmark
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(deterministic)
    matmul_precision = training_cfg.get("matmul_precision")
    if matmul_precision is not None and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(str(matmul_precision))
    return {
        "deterministic": deterministic,
        "cudnn_benchmark": cudnn_benchmark,
        "matmul_precision": getattr(torch, "get_float32_matmul_precision", lambda: None)(),
    }


def move_batch_to_device(batch: Any, device: torch.device, non_blocking: bool) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=non_blocking)
    if isinstance(batch, Mapping):
        return {key: move_batch_to_device(value, device, non_blocking) for key, value in batch.items()}
    if isinstance(batch, tuple):
        return tuple(move_batch_to_device(item, device, non_blocking) for item in batch)
    if isinstance(batch, list):
        return [move_batch_to_device(item, device, non_blocking) for item in batch]
    return batch


def cuda_memory_snapshot() -> dict[str, int | None]:
    if not torch.cuda.is_available():
        return {
            "current_allocated": None,
            "current_reserved": None,
            "peak_allocated": None,
            "peak_reserved": None,
        }
    return {
        "current_allocated": int(torch.cuda.memory_allocated()),
        "current_reserved": int(torch.cuda.memory_reserved()),
        "peak_allocated": int(torch.cuda.max_memory_allocated()),
        "peak_reserved": int(torch.cuda.max_memory_reserved()),
    }
