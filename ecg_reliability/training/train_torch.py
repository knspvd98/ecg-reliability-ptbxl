from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ecg_reliability.data.dataset import CachedECGDataset, manifest_targets
from ecg_reliability.evaluation.metrics import compute_classification_metrics
from ecg_reliability.evaluation.thresholding import sigmoid
from ecg_reliability.models.factory import build_torch_model
from ecg_reliability.performance.dataloader_tuning import dataloader_kwargs_from_config
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
from ecg_reliability.training.checkpoints import save_checkpoint
from ecg_reliability.training.losses import compute_pos_weight, make_bce_loss


def _amp_dtype_from_config(config: dict[str, Any]) -> torch.dtype:
    value = str(config["training"].get("amp_dtype", "float16")).lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    return mapping.get(value, torch.float16)


def _autocast_context(device: torch.device, enabled: bool, amp_dtype: torch.dtype):
    if not enabled or device.type != "cuda":
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, dtype=amp_dtype)
    return torch.cuda.amp.autocast(dtype=amp_dtype)


def _make_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(device="cuda", enabled=enabled)
        except TypeError:
            try:
                return torch.amp.GradScaler("cuda", enabled=enabled)
            except TypeError:
                return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def maybe_compile_model(model: torch.nn.Module, config: dict[str, Any]) -> torch.nn.Module:
    if not bool(config["training"].get("compile_model", False)):
        return model
    if hasattr(torch, "compile"):
        return torch.compile(model)  # type: ignore[attr-defined]
    return model


def build_dataloader(
    manifest_path: str | Path,
    split: str,
    config: dict[str, Any],
    limit: int | None = None,
    shuffle: bool = False,
) -> DataLoader:
    dataset = CachedECGDataset(manifest_path=manifest_path, split=split, limit=limit)
    kwargs = dataloader_kwargs_from_config(config=config, split=split, shuffle=shuffle)
    return DataLoader(dataset, **kwargs)


def run_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: dict[str, Any],
    scaler=None,
    max_batches: int | None = None,
) -> dict[str, Any]:
    model.train()
    meter = ThroughputMeter()
    losses: list[float] = []
    non_blocking = bool(config["training"].get("non_blocking_transfer", False))
    use_amp = bool(config["training"].get("mixed_precision", False)) and device.type == "cuda"
    amp_dtype = _amp_dtype_from_config(config)
    grad_clip_norm = config["training"].get("grad_clip_norm")
    log_every_batches = int(config["training"].get("log_every_batches", 25))
    progress = tqdm(dataloader, desc="train", leave=False)
    for batch_idx, (signals, labels, _) in enumerate(progress):
        if max_batches is not None and batch_idx >= max_batches:
            break
        signals, labels = move_batch_to_device((signals, labels), device, non_blocking=non_blocking)
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, use_amp, amp_dtype):
            logits = model(signals)
            loss = criterion(logits, labels)
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
        meter.update(records=int(signals.shape[0]))
        if (batch_idx + 1) % max(log_every_batches, 1) == 0:
            progress.set_postfix(loss=f"{losses[-1]:.4f}", rps=f"{meter.records_per_second:.1f}")
    summary = meter.summary()
    summary["loss"] = float(np.mean(losses)) if losses else float("nan")
    summary["mixed_precision_active"] = use_amp
    return summary


@torch.no_grad()
def predict_torch_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    config: dict[str, Any] | None = None,
    max_batches: int | None = None,
    return_runtime: bool = False,
) -> dict[str, np.ndarray]:
    model.eval()
    config = config or {"training": {}}
    non_blocking = bool(config["training"].get("non_blocking_transfer", False))
    use_amp = bool(config["training"].get("mixed_precision", False)) and device.type == "cuda"
    amp_dtype = _amp_dtype_from_config(config)
    meter = ThroughputMeter()
    logits_all = []
    labels_all = []
    ecg_ids = []
    for batch_idx, (signals, labels, metadata) in enumerate(tqdm(dataloader, desc="eval", leave=False)):
        if max_batches is not None and batch_idx >= max_batches:
            break
        signals = move_batch_to_device(signals, device, non_blocking=non_blocking)
        with _autocast_context(device, use_amp, amp_dtype):
            logits = model(signals).detach().cpu().numpy()
        logits_all.append(logits)
        labels_all.append(labels.numpy())
        ecg_ids.extend(metadata["ecg_id"].numpy().tolist() if hasattr(metadata["ecg_id"], "numpy") else metadata["ecg_id"])
        meter.update(records=int(signals.shape[0]))
    logits_np = np.concatenate(logits_all, axis=0)
    labels_np = np.concatenate(labels_all, axis=0)
    probs_np = sigmoid(logits_np)
    output = {
        "ecg_ids": np.asarray(ecg_ids, dtype=np.int64),
        "logits": logits_np,
        "probabilities": probs_np,
        "labels": labels_np,
    }
    if return_runtime:
        output["runtime"] = {
            **meter.summary(),
            "device": str(device),
            "mixed_precision_active": use_amp,
        }
    return output


def fit_torch_model(
    config: dict[str, Any],
    manifest_path: str | Path,
    output_dir: str | Path,
    limit_train: int | None = None,
    limit_val: int | None = None,
) -> dict[str, Any]:
    device = resolve_device(config["training"]["device"])
    runtime_flags = configure_torch_runtime(config)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = maybe_compile_model(build_torch_model(config).to(device), config)
    train_loader = build_dataloader(manifest_path, "train", config, limit=limit_train, shuffle=True)
    val_loader = build_dataloader(manifest_path, "val", config, limit=limit_val, shuffle=False)
    pos_weight = None
    if config["training"].get("pos_weight_strategy") == "balanced":
        pos_weight = compute_pos_weight(manifest_targets(manifest_path, "train")).to(device)
    criterion = make_bce_loss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    use_amp = bool(config["training"].get("mixed_precision", False)) and device.type == "cuda"
    scaler = _make_grad_scaler(use_amp)
    best_score = float("-inf")
    best_epoch = 0
    best_path = Path(output_dir) / "best.pt"
    history: list[dict[str, Any]] = []
    stage_timings: dict[str, float] = {}
    epochs_without_improvement = 0
    patience = int(config["training"].get("early_stopping_patience", 0))
    with StageTimer("fit_total", stage_timings):
        for epoch in range(int(config["training"]["epochs"])):
            with StageTimer(f"epoch_{epoch + 1}_train", stage_timings):
                train_stats = run_epoch(model, train_loader, criterion, optimizer, device, config, scaler=scaler)
            with StageTimer(f"epoch_{epoch + 1}_val", stage_timings):
                val_outputs = predict_torch_model(model, val_loader, device, config=config, return_runtime=True)
            val_metrics = compute_classification_metrics(val_outputs["labels"], val_outputs["probabilities"])
            val_score = float(np.nan_to_num(val_metrics["macro_auroc"], nan=-1.0))
            epoch_record = {
                "epoch": epoch + 1,
                "train_loss": train_stats["loss"],
                "train_records_per_second": train_stats["records_per_second"],
                "train_batches_per_second": train_stats["batches_per_second"],
                "val_records_per_second": val_outputs["runtime"]["records_per_second"],
                "val_batches_per_second": val_outputs["runtime"]["batches_per_second"],
                "val_macro_auroc": val_score,
            }
            history.append(epoch_record)
            checkpoint = {
                "epoch": epoch + 1,
                "model_name": config["model"]["name"],
                "config": config,
                "state_dict": model.state_dict(),
                "history": history,
            }
            save_checkpoint(Path(output_dir) / "last.pt", checkpoint)
            if val_score > best_score:
                best_score = val_score
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                save_checkpoint(best_path, checkpoint)
            else:
                epochs_without_improvement += 1
                if patience > 0 and epochs_without_improvement >= patience:
                    break
    runtime_report = {
        **summarize_torch_environment(device),
        "device": str(device),
        "mixed_precision_requested": bool(config["training"].get("mixed_precision", False)),
        "mixed_precision_active": use_amp,
        "amp_dtype": str(_amp_dtype_from_config(config)),
        "batch_size": int(config["training"]["batch_size"]),
        "eval_batch_size": int(config["training"]["eval_batch_size"]),
        "num_workers": int(config["training"]["num_workers"]),
        "best_epoch": int(best_epoch),
        "best_val_macro_auroc": float(best_score),
        "epochs_ran": int(len(history)),
        "epoch_history": history,
        "stage_timings": stage_timings,
        "peak_cuda_memory_allocated": cuda_memory_snapshot()["peak_allocated"],
        "config_training_fields_used": {
            key: config["training"].get(key)
            for key in [
                "device",
                "batch_size",
                "eval_batch_size",
                "num_workers",
                "persistent_workers",
                "prefetch_factor",
                "pin_memory",
                "non_blocking_transfer",
                "epochs",
                "lr",
                "weight_decay",
                "grad_clip_norm",
                "mixed_precision",
                "amp_dtype",
                "early_stopping_patience",
                "deterministic",
                "cudnn_benchmark",
                "matmul_precision",
                "compile_model",
                "log_every_batches",
            ]
        },
        "runtime_flags": runtime_flags,
    }
    runtime_report_path = Path(output_dir) / "training_runtime_report.json"
    write_runtime_report(runtime_report_path, runtime_report)
    return {
        "best_checkpoint": str(best_path),
        "history": history,
        "best_epoch": best_epoch,
        "runtime_report_path": str(runtime_report_path),
    }
