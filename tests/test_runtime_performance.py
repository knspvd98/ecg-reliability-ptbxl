from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ecg_reliability.config import load_config
from ecg_reliability.performance.dataloader_tuning import dataloader_kwargs_from_config
from ecg_reliability.performance.device import configure_torch_runtime
from ecg_reliability.performance.runtime_profile import StageTimer


def test_config_merge_includes_performance_fields() -> None:
    config = load_config("configs/debug.yaml")
    training_cfg = config["training"]
    assert training_cfg["persistent_workers"] is False
    assert training_cfg["prefetch_factor"] is None
    assert training_cfg["pin_memory"] is True
    assert training_cfg["non_blocking_transfer"] is True
    assert training_cfg["deterministic"] is True
    assert training_cfg["cudnn_benchmark"] is False


def test_dataloader_kwargs_with_zero_workers_are_safe() -> None:
    config = {
        "training": {
            "batch_size": 16,
            "eval_batch_size": 32,
            "num_workers": 0,
            "pin_memory": True,
            "persistent_workers": False,
            "prefetch_factor": None,
            "drop_last_train": False,
        }
    }
    kwargs = dataloader_kwargs_from_config(config, split="train", shuffle=True)
    assert kwargs["num_workers"] == 0
    assert "persistent_workers" not in kwargs
    assert "prefetch_factor" not in kwargs


def test_dataloader_kwargs_include_worker_options_when_enabled() -> None:
    config = {
        "training": {
            "batch_size": 16,
            "eval_batch_size": 32,
            "num_workers": 4,
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 2,
            "drop_last_train": False,
        }
    }
    kwargs = dataloader_kwargs_from_config(config, split="train", shuffle=True)
    assert kwargs["num_workers"] == 4
    assert kwargs["persistent_workers"] is True
    assert kwargs["prefetch_factor"] == 2


def test_configure_torch_runtime_respects_flags() -> None:
    import torch

    old_det = torch.backends.cudnn.deterministic
    old_bench = torch.backends.cudnn.benchmark
    try:
        configure_torch_runtime({"training": {"deterministic": True, "cudnn_benchmark": False, "matmul_precision": "high"}})
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
        configure_torch_runtime({"training": {"deterministic": False, "cudnn_benchmark": True, "matmul_precision": "high"}})
        assert torch.backends.cudnn.deterministic is False
        assert torch.backends.cudnn.benchmark is True
    finally:
        torch.backends.cudnn.deterministic = old_det
        torch.backends.cudnn.benchmark = old_bench


def test_stage_timer_records_elapsed_time() -> None:
    timings: dict[str, float] = {}
    with StageTimer("sleep", timings) as timer:
        time.sleep(0.01)
    assert timer.elapsed_seconds > 0.0
    assert timings["sleep"] > 0.0


def test_runtime_preflight_writes_expected_json(tmp_path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    cache_root = tmp_path / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    rows = []
    rng = np.random.default_rng(0)
    idx = 0
    for split, n in [("train", 8), ("val", 4), ("test", 4)]:
        for _ in range(n):
            signal = rng.normal(size=(12, 1000)).astype(np.float32)
            path = cache_root / f"{idx}.npy"
            np.save(path, signal)
            labels = np.zeros(5, dtype=int)
            labels[idx % 5] = 1
            rows.append(
                {
                    "ecg_id": idx,
                    "cache_path": str(path),
                    "split": split,
                    "NORM": int(labels[0]),
                    "MI": int(labels[1]),
                    "STTC": int(labels[2]),
                    "CD": int(labels[3]),
                    "HYP": int(labels[4]),
                }
            )
            idx += 1
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    config_path = tmp_path / "runtime_debug.yaml"
    config_path.write_text(
        "\n".join(
            [
                "data:",
                f"  cache_manifest_path: {manifest_path}",
                "  debug_max_records: 8",
                "training:",
                "  num_workers: 0",
                "  persistent_workers: false",
                "  prefetch_factor: null",
                "  mixed_precision: false",
                "  deterministic: true",
                "  cudnn_benchmark: false",
                "  epochs: 1",
                "model:",
                "  name: simple_cnn",
            ]
        ),
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/11_runtime_preflight.py",
            "--config",
            str(config_path),
            "--model-name",
            "simple_cnn",
            "--max-batches",
            "2",
        ],
        cwd=project_root,
        check=True,
    )
    report_path = project_root / "outputs/summary/runtime_preflight.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for key in [
        "device",
        "cuda_available",
        "model_name",
        "batch_size",
        "num_workers",
        "train_batches_per_second",
        "eval_batches_per_second",
        "recommendations",
    ]:
        assert key in report
    assert report["model_name"] == "simple_cnn"
