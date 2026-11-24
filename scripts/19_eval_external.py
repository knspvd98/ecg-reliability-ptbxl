from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from ecg_reliability.evaluation.model_selection import resolve_calibration_method
from ecg_reliability.evaluation.reports import save_json
from ecg_reliability.external.evaluate_external import evaluate_external_model
from ecg_reliability.runtime import load_runtime_config, model_name_from_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--calibration-method", default="best")
    args = parser.parse_args()

    config = load_runtime_config(args.config, args.override)
    model_name = model_name_from_config(config, args.model_name)
    method = resolve_calibration_method(model_name, args.calibration_method)
    results, frame, sourcewise, overlap_results, runtime_report = evaluate_external_model(
        model_name=model_name,
        config=config,
        calibration_method=method,
    )

    out_dir = Path(f"outputs/external/{model_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "external_binary_results.json", results)
    pd.DataFrame(
        [
            {"score_name": score_name, **metrics}
            for score_name, metrics in results["scores"].items()
        ]
    ).to_csv(out_dir / "external_binary_results.csv", index=False)
    sourcewise.to_csv(out_dir / "external_sourcewise_results.csv", index=False)
    frame.to_csv(out_dir / "external_binary_predictions.csv", index=False)
    if overlap_results is not None:
        save_json(out_dir / "snomed_overlap_results.json", overlap_results)
    save_json(out_dir / "runtime_report.json", runtime_report)
    print(out_dir / "external_binary_results.json")


if __name__ == "__main__":
    main()
