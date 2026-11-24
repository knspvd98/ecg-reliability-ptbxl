from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecg_reliability.runtime import checkpoint_dir, load_runtime_config
from ecg_reliability.training.train_xgboost import fit_xgboost_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    config = load_runtime_config(args.config, args.override)
    output_path = checkpoint_dir("xgboost") / "model.pkl"
    fit_xgboost_model(config, config["data"]["cache_manifest_path"], output_path)
    print(f"Saved XGBoost model to {output_path}")
    print(f"XGBoost runtime report: {checkpoint_dir('xgboost') / 'training_runtime_report.json'}")


if __name__ == "__main__":
    main()
