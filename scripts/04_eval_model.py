from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecg_reliability.evaluation.metrics import compute_classification_metrics
from ecg_reliability.evaluation.reports import save_json, save_metrics_csv, save_prediction_bundle
from ecg_reliability.performance.device import configure_torch_runtime, resolve_device
from ecg_reliability.runtime import (
    checkpoint_dir,
    load_runtime_config,
    load_torch_model,
    model_name_from_config,
    prediction_dir,
)
from ecg_reliability.training.train_torch import build_dataloader, predict_torch_model
from ecg_reliability.training.train_xgboost import predict_xgboost_model


def infer_checkpoint(model_name: str, explicit: str | None) -> Path:
    if explicit is not None:
        return Path(explicit)
    if model_name == "xgboost":
        return checkpoint_dir(model_name) / "model.pkl"
    return checkpoint_dir(model_name) / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    config = load_runtime_config(args.config, args.override)
    model_name = model_name_from_config(config, args.model_name)
    ckpt = infer_checkpoint(model_name, args.checkpoint)
    out_dir = prediction_dir(model_name, "raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    model = None
    device = None
    if model_name != "xgboost":
        device = resolve_device(config["training"]["device"])
        configure_torch_runtime(config)
        model, _ = load_torch_model(ckpt, device=device)

    for split in ("val", "test"):
        if model_name == "xgboost":
            outputs = predict_xgboost_model(ckpt, config["data"]["cache_manifest_path"], split=split)
        else:
            limit = config["data"].get("debug_max_records")
            dataloader = build_dataloader(
                manifest_path=config["data"]["cache_manifest_path"],
                split=split,
                config=config,
                limit=min(limit, 128) if isinstance(limit, int) else None,
                shuffle=False,
            )
            outputs = predict_torch_model(model, dataloader, device=device, config=config)
        metrics = compute_classification_metrics(outputs["labels"], outputs["probabilities"])
        save_prediction_bundle(out_dir, split, outputs["ecg_ids"], outputs["logits"], outputs["probabilities"], outputs["labels"])
        save_json(out_dir / f"{split}_metrics_raw.json", metrics)
        save_metrics_csv(out_dir / f"{split}_metrics_raw.csv", metrics)
        print(f"{model_name} {split}: macro_auroc={metrics['macro_auroc']:.4f} macro_auprc={metrics['macro_auprc']:.4f}")


if __name__ == "__main__":
    main()
