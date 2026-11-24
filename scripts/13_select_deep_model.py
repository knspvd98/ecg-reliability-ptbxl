from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecg_reliability.evaluation.model_selection import select_best_deep_model
from ecg_reliability.evaluation.reports import save_json
from ecg_reliability.runtime import load_runtime_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--candidates", nargs="+", required=True)
    args = parser.parse_args()

    load_runtime_config(args.config, args.override)
    payload = select_best_deep_model("outputs/predictions", args.candidates)
    save_json("outputs/summary/model_selection.json", payload)
    print(f"Selected deep model: {payload['selected_deep_model']}")
    print("outputs/summary/model_selection.json")


if __name__ == "__main__":
    main()
