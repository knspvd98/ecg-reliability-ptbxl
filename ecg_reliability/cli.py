from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ecg_reliability.config import apply_overrides, load_config
from ecg_reliability.data.cache import compute_preprocessing_fingerprint, load_cache_provenance
from ecg_reliability.experiments.registry import DEFAULT_RUNS_ROOT, list_runs, load_run_manifest
from ecg_reliability.experiments.runner import (
    ALL_SUITES,
    generate_synthetic_ecg_cohort,
    run_benchmark_comparison,
    run_model_experiment,
)


def _cmd_run_suite(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.override:
        config = apply_overrides(config, args.override)

    suites = tuple(ALL_SUITES) if args.suites == "all" else tuple(s.strip() for s in args.suites.split(",") if s.strip())
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    runs_root = Path(args.runs_root)

    ds_prov: dict | None = None
    manifest_path = args.manifest
    if args.tier == "synthetic_smoke" and manifest_path is None:
        cohort = generate_synthetic_ecg_cohort(
            output_root=Path("outputs/cache/synthetic_benchmark"),
            n_train_patients=args.synthetic_train_patients,
            n_val_patients=args.synthetic_val_patients,
            n_test_patients=args.synthetic_test_patients,
            records_per_patient=2,
            sampling_rate=int(config["data"]["sampling_rate"]),
            signal_length=int(config["data"]["signal_length"]),
            normalize=bool(config["data"].get("normalize_per_lead", True)),
            seed=int(config["project"]["seed"]),
        )
        manifest_path = cohort["manifest_path"]
        ds_prov = cohort

    if manifest_path is None:
        manifest_path = config["data"]["cache_manifest_path"]
    if not Path(manifest_path).exists():
        raise FileNotFoundError(f"Cache manifest not found: {manifest_path}")

    if args.resume:
        if len(models) != 1:
            raise ValueError("--resume requires exactly one model.")
        manifest = run_model_experiment(
            config=config,
            model_name=models[0],
            manifest_path=manifest_path,
            execution_tier=args.tier,
            suites=suites,
            runs_root=runs_root,
            resume_run_id=args.resume,
            dataset_provenance=ds_prov,
            n_bootstrap=args.n_bootstrap,
            robustness_limit=args.robustness_limit,
            xai_samples=args.xai_samples,
        )
        print(json.dumps({"run_id": manifest["run_id"], "status": manifest["status"]}, indent=2))
        return 0

    if len(models) == 1:
        manifest = run_model_experiment(
            config=config,
            model_name=models[0],
            manifest_path=manifest_path,
            execution_tier=args.tier,
            suites=suites,
            runs_root=runs_root,
            dataset_provenance=ds_prov,
            n_bootstrap=args.n_bootstrap,
            robustness_limit=args.robustness_limit,
            xai_samples=args.xai_samples,
        )
        print(json.dumps({"run_id": manifest["run_id"], "status": manifest["status"]}, indent=2))
        return 0

    summary = run_benchmark_comparison(
        base_config=config,
        models=models,
        manifest_path=manifest_path,
        execution_tier=args.tier,
        suites=suites,
        runs_root=runs_root,
        dataset_provenance=ds_prov,
        n_bootstrap=args.n_bootstrap,
        robustness_limit=args.robustness_limit,
        xai_samples=args.xai_samples,
    )
    print(json.dumps(summary["completed_run_ids"], indent=2))
    return 0


def _cmd_list_runs(args: argparse.Namespace) -> int:
    runs = list_runs(args.runs_root)
    if args.json:
        print(json.dumps(runs, indent=2))
        return 0
    if not runs:
        print(f"No runs found in {args.runs_root}")
        return 0
    print(f"{'RUN_ID':<52} {'STATUS':<11} {'TIER':<16} {'MODEL':<14} {'AUROC':>7} {'ECE':>7}")
    print("-" * 112)
    for run in runs:
        raw = (run.get("metrics") or {}).get("raw_test") or {}
        cal = (run.get("metrics") or {}).get("calibrated_test") or {}
        auroc = raw.get("macro_auroc")
        ece = cal.get("macro_ece")
        auroc_s = f"{auroc:.4f}" if isinstance(auroc, (int, float)) else "n/a"
        ece_s = f"{ece:.4f}" if isinstance(ece, (int, float)) else "n/a"
        model = (run.get("model_identity") or {}).get("model_name", "?")
        print(f"{run['run_id']:<52} {str(run.get('status')):<11} {str(run.get('execution_tier')):<16} {model:<14} {auroc_s:>7} {ece_s:>7}")
    return 0


def _cmd_inspect_run(args: argparse.Namespace) -> int:
    manifest = load_run_manifest(args.run_id, runs_root=args.runs_root)
    print(json.dumps(manifest, indent=2))
    return 0


def _cmd_validate_cache(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.override:
        config = apply_overrides(config, args.override)
    manifest_path = Path(config["data"]["cache_manifest_path"])
    prov_path = manifest_path.parent / "cache_provenance.json"
    expected_fp = compute_preprocessing_fingerprint(config)
    provenance = load_cache_provenance(prov_path)
    report = {
        "manifest_exists": manifest_path.exists(),
        "provenance_exists": prov_path.exists(),
        "expected_preprocessing_fingerprint": expected_fp,
        "observed_preprocessing_fingerprint": provenance.get("preprocessing_fingerprint") if provenance else None,
        "fingerprint_match": bool(provenance and provenance.get("preprocessing_fingerprint") == expected_fp),
        "n_provenance_records": len(provenance.get("records", {})) if provenance else 0,
    }
    print(json.dumps(report, indent=2))
    return 0 if report["fingerprint_match"] else 1


def _cmd_workbench(args: argparse.Namespace) -> int:
    from ecg_reliability.workbench.server import serve_workbench

    serve_workbench(
        host=args.host,
        port=args.port,
        runs_root=Path(args.runs_root),
        outputs_root=Path(args.outputs_root),
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ecg-reliability",
        description="ECG Reliability Research Platform CLI (research software only — not a medical device).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run-suite", help="Execute a provenance-tracked ECG reliability experiment suite.")
    p_run.add_argument("--config", default="configs/debug.yaml")
    p_run.add_argument("--override", action="append", default=[])
    p_run.add_argument("--models", default="xgboost,simple_cnn,resnet1d,inceptiontime,tcn")
    p_run.add_argument("--tier", choices=["synthetic_smoke", "bounded_real", "full_campaign"], default="synthetic_smoke")
    p_run.add_argument("--suites", default="all", help="Comma-separated suites or 'all'.")
    p_run.add_argument("--manifest", default=None, help="Explicit cache manifest path.")
    p_run.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    p_run.add_argument("--resume", default=None, help="Resume an interrupted/failed run_id.")
    p_run.add_argument("--n-bootstrap", type=int, default=100)
    p_run.add_argument("--robustness-limit", type=int, default=32)
    p_run.add_argument("--xai-samples", type=int, default=4)
    p_run.add_argument("--synthetic-train-patients", type=int, default=36)
    p_run.add_argument("--synthetic-val-patients", type=int, default=12)
    p_run.add_argument("--synthetic-test-patients", type=int, default=16)
    p_run.set_defaults(func=_cmd_run_suite)

    p_list = sub.add_parser("list-runs", help="List recorded experiment runs and their statuses.")
    p_list.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    p_list.add_argument("--json", action="store_true")
    p_list.set_defaults(func=_cmd_list_runs)

    p_inspect = sub.add_parser("inspect-run", help="Print the full provenance manifest for a run.")
    p_inspect.add_argument("run_id")
    p_inspect.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    p_inspect.set_defaults(func=_cmd_inspect_run)

    p_cache = sub.add_parser("validate-cache", help="Validate cache provenance against current preprocessing config.")
    p_cache.add_argument("--config", default="configs/base.yaml")
    p_cache.add_argument("--override", action="append", default=[])
    p_cache.set_defaults(func=_cmd_validate_cache)

    p_wb = sub.add_parser("workbench", help="Launch the local interactive ECG Research Workbench.")
    p_wb.add_argument("--host", default="127.0.0.1")
    p_wb.add_argument("--port", type=int, default=8501)
    p_wb.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    p_wb.add_argument("--outputs-root", default="outputs")
    p_wb.set_defaults(func=_cmd_workbench)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
