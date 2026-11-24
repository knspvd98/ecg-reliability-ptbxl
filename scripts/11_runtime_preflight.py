from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from ecg_reliability.data.dataset import manifest_targets
from ecg_reliability.evaluation.reports import save_json
from ecg_reliability.models.factory import build_torch_model
from ecg_reliability.performance.dataloader_tuning import recommend_runtime_settings
from ecg_reliability.performance.device import configure_torch_runtime, cuda_memory_snapshot, resolve_device
from ecg_reliability.performance.runtime_profile import StageTimer, summarize_torch_environment
from ecg_reliability.runtime import load_runtime_config, model_name_from_config
from ecg_reliability.training.losses import compute_pos_weight, make_bce_loss
from ecg_reliability.training.train_torch import build_dataloader, maybe_compile_model, predict_torch_model, run_epoch


def resolve_manifest_path(config: dict) -> str:
    manifest_path = Path(config["data"]["cache_manifest_path"])
    if manifest_path.exists():
        return str(manifest_path)
    synthetic_manifest = Path("outputs/cache/synthetic_smoke_manifest.csv")
    if synthetic_manifest.exists():
        return str(synthetic_manifest)
    raise FileNotFoundError(
        f"Missing cache manifest: {manifest_path}. Run cache preparation or synthetic smoke first."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--max-batches", type=int, default=10)
    args = parser.parse_args()

    config = load_runtime_config(args.config, args.override)
    model_name = model_name_from_config(config, args.model_name)
    config["model"]["name"] = model_name
    manifest_path = resolve_manifest_path(config)
    device = resolve_device(config["training"]["device"])
    runtime_flags = configure_torch_runtime(config)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    train_loader = build_dataloader(manifest_path, "train", config, limit=config["data"].get("debug_max_records"), shuffle=True)
    val_limit = config["data"].get("debug_max_records")
    val_loader = build_dataloader(manifest_path, "val", config, limit=val_limit, shuffle=False)
    model = maybe_compile_model(build_torch_model(config).to(device), config)
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
    scaler = None
    if use_amp:
        from ecg_reliability.training.train_torch import _make_grad_scaler

        scaler = _make_grad_scaler(True)

    timings: dict[str, float] = {}
    with StageTimer("preflight_train", timings):
        train_stats = run_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            config=config,
            scaler=scaler,
            max_batches=args.max_batches,
        )
    with StageTimer("preflight_eval", timings):
        eval_outputs = predict_torch_model(
            model=model,
            dataloader=val_loader,
            device=device,
            config=config,
            max_batches=args.max_batches,
            return_runtime=True,
        )

    report = {
        **summarize_torch_environment(device),
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "model_name": model_name,
        "manifest_path_used": manifest_path,
        "batch_size": int(config["training"]["batch_size"]),
        "num_workers": int(config["training"]["num_workers"]),
        "mixed_precision_requested": bool(config["training"].get("mixed_precision", False)),
        "mixed_precision_active": use_amp,
        "train_batches_per_second": float(train_stats["batches_per_second"]),
        "train_records_per_second": float(train_stats["records_per_second"]),
        "eval_batches_per_second": float(eval_outputs["runtime"]["batches_per_second"]),
        "eval_records_per_second": float(eval_outputs["runtime"]["records_per_second"]),
        "peak_cuda_memory": cuda_memory_snapshot()["peak_allocated"],
        "cpu_fallback_warning": bool(device.type == "cpu"),
        "low_throughput_warning": bool(torch.cuda.is_available() and device.type == "cuda" and train_stats["records_per_second"] < 50.0),
        "stage_timings": timings,
        "runtime_flags": runtime_flags,
    }
    report["recommendations"] = recommend_runtime_settings(config, report)
    save_json("outputs/summary/runtime_preflight.json", report)
    print("outputs/summary/runtime_preflight.json")


if __name__ == "__main__":
    main()
