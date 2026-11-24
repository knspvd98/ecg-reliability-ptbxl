from __future__ import annotations

import os
import platform
import time
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import torch

from ecg_reliability.evaluation.reports import save_json


class StageTimer(AbstractContextManager):
    def __init__(self, stage_name: str, store: dict[str, float] | None = None) -> None:
        self.stage_name = stage_name
        self.store = store
        self.elapsed_seconds = 0.0
        self._start = 0.0

    def __enter__(self) -> StageTimer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        self.elapsed_seconds = time.perf_counter() - self._start
        if self.store is not None:
            self.store[self.stage_name] = self.elapsed_seconds
        return None


class ThroughputMeter:
    def __init__(self) -> None:
        self._start = time.perf_counter()
        self.records = 0
        self.batches = 0

    def update(self, records: int, batches: int = 1) -> None:
        self.records += int(records)
        self.batches += int(batches)

    @property
    def elapsed_seconds(self) -> float:
        return max(time.perf_counter() - self._start, 1e-9)

    @property
    def records_per_second(self) -> float:
        return float(self.records / self.elapsed_seconds)

    @property
    def batches_per_second(self) -> float:
        return float(self.batches / self.elapsed_seconds)

    def summary(self) -> dict[str, float | int]:
        return {
            "records": int(self.records),
            "batches": int(self.batches),
            "elapsed_seconds": float(self.elapsed_seconds),
            "records_per_second": float(self.records_per_second),
            "batches_per_second": float(self.batches_per_second),
        }


def summarize_torch_environment(current_device: str | torch.device | None = None) -> dict[str, Any]:
    device = torch.device(current_device) if current_device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matmul_precision = None
    if hasattr(torch, "get_float32_matmul_precision"):
        matmul_precision = torch.get_float32_matmul_precision()
    cuda_available = bool(torch.cuda.is_available())
    cuda_device_name = torch.cuda.get_device_name(device) if cuda_available and device.type == "cuda" else None
    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_device_name": cuda_device_name,
        "cuda_device_count": int(torch.cuda.device_count()) if cuda_available else 0,
        "current_device": str(device),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "matmul_precision": matmul_precision,
        "cpu_count": int(os.cpu_count() or 0),
        "process_pid": int(os.getpid()),
    }


def write_runtime_report(path: str | Path, payload: dict[str, Any]) -> None:
    save_json(path, payload)
