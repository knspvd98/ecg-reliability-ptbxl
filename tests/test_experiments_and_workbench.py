from __future__ import annotations

import json
import threading
import urllib.request

import numpy as np
import pytest

from ecg_reliability.config import load_config
from ecg_reliability.experiments.registry import RunOverwriteError, create_run, mark_run_completed, mark_run_failed
from ecg_reliability.experiments.runner import (
    generate_synthetic_ecg_cohort,
    run_benchmark_comparison,
    run_model_experiment,
)
from ecg_reliability.workbench.server import create_workbench_server


def _http_json(url: str, payload: dict | None = None) -> dict:
    if payload is None:
        with urllib.request.urlopen(url, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def test_registry_prevents_overwriting_completed_runs_and_records_failures(tmp_path) -> None:
    config = load_config("configs/debug.yaml")
    runs_root = tmp_path / "runs"

    manifest, _ = create_run(
        config=config,
        model_name="simple_cnn",
        execution_tier="synthetic_smoke",
        run_id="fixed_test_run",
        runs_root=runs_root,
    )
    mark_run_completed(manifest, runs_root=runs_root)

    with pytest.raises(RunOverwriteError, match="Refusing to overwrite completed run"):
        create_run(
            config=config,
            model_name="simple_cnn",
            execution_tier="synthetic_smoke",
            run_id="fixed_test_run",
            runs_root=runs_root,
        )

    failed_manifest, _ = create_run(
        config=config,
        model_name="resnet1d",
        execution_tier="synthetic_smoke",
        run_id="failed_test_run",
        runs_root=runs_root,
    )
    mark_run_failed(failed_manifest, stage_name="training", exc=RuntimeError("simulated OOM"), runs_root=runs_root)
    assert failed_manifest["status"] == "failed"
    assert failed_manifest["failure_info"]["stage"] == "training"


def test_end_to_end_benchmark_and_live_workbench_api(tmp_path) -> None:
    cohort = generate_synthetic_ecg_cohort(
        tmp_path / "synthetic",
        n_train_patients=10,
        n_val_patients=5,
        n_test_patients=6,
        records_per_patient=2,
        seed=42,
    )
    assert cohort["patient_overlap_report"]["has_patient_leakage"] is False

    config = load_config("configs/debug.yaml")
    config["training"]["epochs"] = 1
    config["training"]["batch_size"] = 8
    config["training"]["eval_batch_size"] = 8
    config["training"]["num_workers"] = 0
    config["xgboost"]["n_estimators"] = 10
    config["xgboost"]["n_jobs"] = 1
    runs_root = tmp_path / "runs"

    comp = run_benchmark_comparison(
        base_config=config,
        models=["xgboost", "simple_cnn"],
        manifest_path=cohort["manifest_path"],
        execution_tier="synthetic_smoke",
        runs_root=runs_root,
        dataset_provenance=cohort,
        n_bootstrap=25,
        robustness_limit=8,
        xai_samples=2,
    )
    assert set(comp["completed_run_ids"].keys()) == {"xgboost", "simple_cnn"}
    cnn_run_id = comp["completed_run_ids"]["simple_cnn"]

    # Attempting to overwrite completed run fails cleanly
    with pytest.raises(RunOverwriteError):
        run_model_experiment(
            config=config,
            model_name="simple_cnn",
            manifest_path=cohort["manifest_path"],
            execution_tier="synthetic_smoke",
            runs_root=runs_root,
            run_id=cnn_run_id,
        )

    # Start real Workbench HTTP server on ephemeral port and verify every workspace API
    server = create_workbench_server(host="127.0.0.1", port=0, runs_root=runs_root, outputs_root=tmp_path)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    try:
        health = _http_json(f"{base_url}/api/health")
        assert health["status"] == "ok"
        assert health["medical_device"] is False

        runs_resp = _http_json(f"{base_url}/api/runs")
        assert len(runs_resp["runs"]) == 2

        detail = _http_json(f"{base_url}/api/runs/{cnn_run_id}")
        assert detail["manifest"]["status"] == "completed"
        assert len(detail["files"]) > 5

        signals = _http_json(f"{base_url}/api/runs/{cnn_run_id}/signals?split=test&limit=10")
        assert len(signals["records"]) > 0
        ecg_id = signals["records"][0]["ecg_id"]

        sig_detail = _http_json(f"{base_url}/api/runs/{cnn_run_id}/signal/{ecg_id}")
        assert len(sig_detail["signal"]) == 12
        assert sig_detail["sqi"]["num_leads"] == 12

        pert = _http_json(
            f"{base_url}/api/runs/{cnn_run_id}/perturb",
            {"ecg_id": ecg_id, "perturbation": "baseline_wander", "params": {"amplitude": 0.4, "frequency_hz": 0.33}},
        )
        assert len(pert["perturbed_signal"]) == 12
        assert pert["clean_prediction"] is not None
        assert pert["perturbed_prediction"] is not None

        rel = _http_json(f"{base_url}/api/runs/{cnn_run_id}/reliability")
        assert rel["reliability_bins"] is not None
        assert rel["bootstrap_ci"] is not None

        raw_test_dir = runs_root / cnn_run_id / "predictions" / "raw" / "test"
        n_test = len(np.load(raw_test_dir / "labels.npy"))
        rel_adaptive = _http_json(f"{base_url}/api/runs/{cnn_run_id}/reliability?method=raw&binning=adaptive")
        assert rel_adaptive["binning"] == "adaptive"
        assert rel_adaptive["active_calibration_method"] == "raw"
        for bins in rel_adaptive["reliability_bins"]["per_class"].values():
            assert sum(b["count"] for b in bins["raw"]) == n_test
            assert bins["raw"] == bins["calibrated"]

        # Verify redesigned workbench shell & static assets serve cleanly without numbered tabs or badge bloat
        with urllib.request.urlopen(f"{base_url}/", timeout=10) as html_resp:
            html = html_resp.read().decode("utf-8")
        assert "Research software only — not a medical device" in html
        assert "1. Signal Explorer" not in html
        assert "SYNTHETIC SMOKE DATA" not in html
        for asset in ("/styles.css", "/ecg_canvas.js", "/charts.js", "/app.js"):
            with urllib.request.urlopen(f"{base_url}{asset}", timeout=10) as asset_resp:
                assert asset_resp.status == 200
                assert len(asset_resp.read()) > 500

        rob = _http_json(f"{base_url}/api/runs/{cnn_run_id}/robustness")
        assert rob["available"] is True
        assert len(rob["results"]["delta_summary"]) >= 10

        xai = _http_json(f"{base_url}/api/runs/{cnn_run_id}/explainability?ecg_id={ecg_id}&target_label=MI&method=ig")
        assert xai["selected_attribution"] is not None
        assert len(xai["selected_attribution"]["lead_importance"]) == 12
        assert xai["failure_analysis"] is not None

        occlusion_path = runs_root / cnn_run_id / "explanations" / f"ecg_{ecg_id}_mi_occlusion.npy"
        occlusion_path.unlink(missing_ok=True)
        occlusion = _http_json(f"{base_url}/api/runs/{cnn_run_id}/explainability?ecg_id={ecg_id}&target_label=MI&method=occlusion")
        attribution = occlusion["selected_attribution"]
        assert attribution["method"] == "occlusion"
        assert attribution["source"] == "computed_on_demand_from_checkpoint"
        assert not np.allclose(attribution["attribution"], xai["selected_attribution"]["attribution"])

        (runs_root / cnn_run_id / "predictions" / "calibrated" / "platt" / "test" / "labels.npy").unlink()
        missing = _http_json(f"{base_url}/api/runs/{cnn_run_id}/reliability?method=platt&binning=adaptive")
        assert missing["active_calibration_method"] == "platt"
        assert missing["confidence_distributions"] == {}
        assert missing["reliability_bins"]["per_class"]["MI"]["raw"]
        assert missing["reliability_bins"]["per_class"]["MI"]["calibrated"] is None

        comps_api = _http_json(f"{base_url}/api/comparisons")
        assert comps_api["paired_comparisons"] is not None
        assert comps_api["model_disagreement"]["shared_ecg_count"] > 0
    finally:
        server.shutdown()
        server.server_close()
