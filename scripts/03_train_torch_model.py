from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecg_reliability.runtime import checkpoint_dir, load_runtime_config, model_name_from_config
from ecg_reliability.training.train_torch import fit_torch_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--model-name", default=None)
    args = parser.parse_args()

    config = load_runtime_config(args.config, args.override)
    model_name = model_name_from_config(config, args.model_name)
    config["model"]["name"] = model_name
    limit = config["data"].get("debug_max_records")
    result = fit_torch_model(
        config=config,
        manifest_path=config["data"]["cache_manifest_path"],
        output_dir=checkpoint_dir(model_name),
        limit_train=limit,
        limit_val=min(limit, 128) if isinstance(limit, int) else None,
    )
    print(f"Finished training {model_name}: {result['best_checkpoint']}")


if __name__ == "__main__":
    main()
