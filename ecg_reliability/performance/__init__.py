"""Runtime performance helpers for device, timing, and preflight checks."""

from ecg_reliability.performance.dataloader_tuning import dataloader_kwargs_from_config, recommend_runtime_settings
from ecg_reliability.performance.device import (
    configure_torch_runtime,
    cuda_memory_snapshot,
    move_batch_to_device,
    resolve_device,
)
from ecg_reliability.performance.runtime_profile import (
    StageTimer,
    ThroughputMeter,
    summarize_torch_environment,
    write_runtime_report,
)

__all__ = [
    "StageTimer",
    "ThroughputMeter",
    "configure_torch_runtime",
    "cuda_memory_snapshot",
    "dataloader_kwargs_from_config",
    "move_batch_to_device",
    "recommend_runtime_settings",
    "resolve_device",
    "summarize_torch_environment",
    "write_runtime_report",
]
