from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ecg_reliability.data.cache import atomic_write_json, compute_preprocessing_fingerprint
from ecg_reliability.paths import project_root

EXECUTION_TIERS = ("synthetic_smoke", "bounded_real", "full_campaign")
DEFAULT_RUNS_ROOT = Path("outputs/runs")


class RunOverwriteError(RuntimeError):
    """Raised when attempting to overwrite an already-completed research run."""


def file_sha256(path: str | Path) -> str | None:
    target = Path(path)
    if not target.exists() or not target.is_file():
        return None
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_code_version() -> dict[str, Any]:
    root = project_root()
    commit = None
    branch = None
    dirty = None
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(status)
    except Exception:
        pass
    return {
        "git_commit": commit,
        "git_branch": branch,
        "git_dirty": dirty,
    }


def get_hardware_software_metadata() -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for pkg in ("numpy", "scipy", "pandas", "sklearn", "torch", "xgboost", "captum", "wfdb", "matplotlib"):
        try:
            mod = __import__(pkg)
            versions[pkg] = str(getattr(mod, "__version__", "unknown"))
        except Exception:
            versions[pkg] = None

    cuda_available = False
    cuda_device_name = None
    mps_available = False
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            cuda_device_name = torch.cuda.get_device_name(0)
        mps_available = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    except Exception:
        pass

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python_version": platform.python_version(),
        "cuda_available": cuda_available,
        "cuda_device_name": cuda_device_name,
        "mps_available": mps_available,
        "packages": versions,
    }


def generate_run_id(execution_tier: str, model_name: str, suite_name: str = "suite") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    short = uuid.uuid4().hex[:6]
    safe_tier = execution_tier.replace("_", "-")
    safe_model = model_name.replace("_", "-")
    safe_suite = suite_name.replace("_", "-")
    return f"{stamp}_{safe_tier}_{safe_model}_{safe_suite}_{short}"


def _index_path(runs_root: str | Path = DEFAULT_RUNS_ROOT) -> Path:
    return Path(runs_root) / "index.json"


def _manifest_path(run_id: str, runs_root: str | Path = DEFAULT_RUNS_ROOT) -> Path:
    return Path(runs_root) / run_id / "run_manifest.json"


def load_run_manifest(run_id: str, runs_root: str | Path = DEFAULT_RUNS_ROOT) -> dict[str, Any]:
    path = _manifest_path(run_id, runs_root)
    if not path.exists():
        raise FileNotFoundError(f"Run manifest not found for run_id={run_id}: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _update_run_index(manifest: dict[str, Any], runs_root: str | Path = DEFAULT_RUNS_ROOT) -> None:
    root = Path(runs_root)
    root.mkdir(parents=True, exist_ok=True)
    idx_file = _index_path(root)
    index: dict[str, Any] = {"runs": {}}
    if idx_file.exists():
        try:
            with idx_file.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict) and isinstance(loaded.get("runs"), dict):
                index = loaded
        except (OSError, json.JSONDecodeError):
            index = {"runs": {}}

    run_id = str(manifest["run_id"])
    raw_metrics = (manifest.get("metrics") or {}).get("raw_test") or {}
    cal_metrics = (manifest.get("metrics") or {}).get("calibrated_test") or {}
    index["runs"][run_id] = {
        "run_id": run_id,
        "status": manifest.get("status"),
        "execution_tier": manifest.get("execution_tier"),
        "suite_name": manifest.get("suite_name"),
        "model_name": (manifest.get("model_identity") or {}).get("model_name"),
        "created_at_utc": manifest.get("created_at_utc"),
        "updated_at_utc": manifest.get("updated_at_utc"),
        "macro_auroc": raw_metrics.get("macro_auroc"),
        "macro_auprc": raw_metrics.get("macro_auprc"),
        "macro_ece": cal_metrics.get("macro_ece"),
        "macro_brier": cal_metrics.get("macro_brier"),
        "manifest_path": str(_manifest_path(run_id, root)),
    }
    index["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(idx_file, index)


def save_run_manifest(manifest: dict[str, Any], runs_root: str | Path = DEFAULT_RUNS_ROOT) -> Path:
    manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    path = _manifest_path(str(manifest["run_id"]), runs_root)
    atomic_write_json(path, manifest)
    _update_run_index(manifest, runs_root)
    return path


def create_run(
    *,
    config: dict[str, Any],
    model_name: str,
    execution_tier: str,
    suite_name: str = "full_reliability",
    dataset_provenance: dict[str, Any] | None = None,
    model_identity: dict[str, Any] | None = None,
    run_id: str | None = None,
    runs_root: str | Path = DEFAULT_RUNS_ROOT,
) -> tuple[dict[str, Any], Path]:
    """Initialize an isolated research run directory with full provenance metadata."""
    if execution_tier not in EXECUTION_TIERS:
        raise ValueError(f"Unsupported execution_tier={execution_tier!r}; expected one of {EXECUTION_TIERS}")

    rid = run_id or generate_run_id(execution_tier, model_name, suite_name)
    run_dir = Path(runs_root) / rid
    manifest_file = run_dir / "run_manifest.json"
    if manifest_file.exists():
        existing = load_run_manifest(rid, runs_root)
        if existing.get("status") == "completed":
            raise RunOverwriteError(
                f"Refusing to overwrite completed run {rid!r}. Create a new run_id to preserve provenance."
            )

    run_dir.mkdir(parents=True, exist_ok=True)
    # Persist YAML config snapshot for exact reproducibility
    config_snapshot_path = run_dir / "config_snapshot.yaml"
    config_snapshot_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    now = datetime.now(timezone.utc).isoformat()
    manifest: dict[str, Any] = {
        "run_id": rid,
        "status": "running",
        "execution_tier": execution_tier,
        "scientific_claim_eligible": bool(execution_tier == "full_campaign"),
        "suite_name": suite_name,
        "created_at_utc": now,
        "updated_at_utc": now,
        "dataset_provenance": dataset_provenance or {},
        "config_snapshot": config,
        "config_snapshot_path": str(config_snapshot_path),
        "seeds": {
            "project_seed": int(config.get("project", {}).get("seed", 42)),
            "deterministic_torch": bool(config.get("training", {}).get("deterministic", False)),
        },
        "code_version": get_code_version(),
        "model_identity": {
            "model_name": model_name,
            **(model_identity or {}),
        },
        "preprocessing_fingerprint": compute_preprocessing_fingerprint(config),
        "hardware_software_metadata": get_hardware_software_metadata(),
        "checkpoints": {},
        "stages": {},
        "metrics": {},
        "plots": [],
        "artifacts": {},
        "failure_info": None,
    }
    save_run_manifest(manifest, runs_root)
    return manifest, run_dir


def prepare_resumable_run(run_id: str, runs_root: str | Path = DEFAULT_RUNS_ROOT) -> tuple[dict[str, Any], Path]:
    """Load an interrupted or failed run for recovery while refusing completed runs."""
    manifest = load_run_manifest(run_id, runs_root)
    if manifest.get("status") == "completed":
        raise RunOverwriteError(f"Run {run_id!r} is already completed and cannot be overwritten or re-run.")
    manifest["status"] = "running"
    manifest["resumed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["failure_info"] = None
    save_run_manifest(manifest, runs_root)
    return manifest, Path(runs_root) / run_id


def mark_stage_status(
    manifest: dict[str, Any],
    stage_name: str,
    status: str,
    *,
    details: dict[str, Any] | None = None,
    runs_root: str | Path = DEFAULT_RUNS_ROOT,
) -> None:
    manifest.setdefault("stages", {})[stage_name] = {
        "status": status,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **(details or {}),
    }
    save_run_manifest(manifest, runs_root)


def mark_run_failed(
    manifest: dict[str, Any],
    *,
    stage_name: str,
    exc: BaseException,
    runs_root: str | Path = DEFAULT_RUNS_ROOT,
) -> None:
    status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
    manifest["status"] = status
    manifest["failure_info"] = {
        "stage": stage_name,
        "exception_type": exc.__class__.__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
        "failed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest.setdefault("stages", {})[stage_name] = {
        "status": status,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "error": f"{exc.__class__.__name__}: {exc}",
    }
    save_run_manifest(manifest, runs_root)


def mark_run_completed(
    manifest: dict[str, Any],
    runs_root: str | Path = DEFAULT_RUNS_ROOT,
) -> None:
    manifest["status"] = "completed"
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["failure_info"] = None
    save_run_manifest(manifest, runs_root)


def list_runs(runs_root: str | Path = DEFAULT_RUNS_ROOT) -> list[dict[str, Any]]:
    """Return all recorded runs (completed, failed, interrupted, running), newest first."""
    root = Path(runs_root)
    if not root.exists():
        return []
    runs: list[dict[str, Any]] = []
    for manifest_file in root.glob("*/run_manifest.json"):
        try:
            with manifest_file.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            runs.append(data)
        except (OSError, json.JSONDecodeError):
            continue
    runs.sort(key=lambda item: str(item.get("created_at_utc", "")), reverse=True)
    return runs
