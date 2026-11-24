from __future__ import annotations

from typing import Any


def dataloader_kwargs_from_config(config: dict[str, Any], split: str, shuffle: bool) -> dict[str, Any]:
    training_cfg = config["training"]
    batch_size_key = "batch_size" if split == "train" else "eval_batch_size"
    num_workers = int(training_cfg["num_workers"])
    kwargs: dict[str, Any] = {
        "batch_size": int(training_cfg[batch_size_key]),
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": bool(training_cfg.get("pin_memory", False)),
        "drop_last": bool(training_cfg.get("drop_last_train", False)) if split == "train" else False,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(training_cfg.get("persistent_workers", False))
        prefetch_factor = training_cfg.get("prefetch_factor")
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(prefetch_factor)
    return kwargs


def recommend_runtime_settings(config: dict[str, Any], observed_report: dict[str, Any]) -> list[str]:
    recommendations: list[str] = []
    training_cfg = config["training"]
    num_workers = int(training_cfg.get("num_workers", 0))
    if num_workers == 0 and training_cfg.get("persistent_workers", False):
        recommendations.append("disable persistent_workers when num_workers=0")
    if observed_report.get("cuda_available") and observed_report.get("device") == "cpu":
        recommendations.append("CUDA is available but runtime fell back to CPU; check device selection")
    if observed_report.get("cuda_available") and not observed_report.get("mixed_precision_active", False) and training_cfg.get("mixed_precision", False):
        recommendations.append("mixed_precision was requested but not active; check AMP device compatibility")
    train_records_sec = float(observed_report.get("train_records_per_second", 0.0) or 0.0)
    if observed_report.get("cuda_available") and observed_report.get("device") != "cpu" and train_records_sec < 50.0:
        recommendations.append("check cache I/O or num_workers; GPU throughput looks low")
    if not observed_report.get("cuda_available") and training_cfg.get("mixed_precision", False):
        recommendations.append("enable mixed_precision only on CUDA runs")
    if num_workers < 2 and observed_report.get("device") not in {None, "cpu"}:
        recommendations.append("increase num_workers if data loading becomes a bottleneck")
    if int(training_cfg.get("batch_size", 0)) <= 8 and observed_report.get("device") not in {None, "cpu"}:
        recommendations.append("increase batch_size if GPU memory allows")
    return recommendations
