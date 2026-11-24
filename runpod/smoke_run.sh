#!/usr/bin/env bash
set -euo pipefail

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import json
import numpy as np
import pandas as pd

root = Path("outputs/cache/synthetic_smoke")
root.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)
rows = []
idx = 0
for split, n in [("train", 32), ("val", 8), ("test", 8)]:
    for _ in range(n):
        signal = rng.normal(0, 1, size=(12, 1000)).astype("float32")
        labels = np.zeros(5, dtype=int)
        active = rng.choice(5, size=rng.integers(1, 3), replace=False)
        labels[active] = 1
        for class_idx in active:
            signal[class_idx % 12, 100:200] += 1.5 + class_idx * 0.2
        path = root / f"{idx}.npy"
        np.save(path, signal)
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
manifest_path = Path("outputs/cache/synthetic_smoke_manifest.csv")
pd.DataFrame(rows).to_csv(manifest_path, index=False)
summary = {
    "status": "synthetic_smoke_prepared",
    "n_records": len(rows),
    "manifest_path": str(manifest_path),
    "cache_root": str(root),
}
Path("outputs/summary").mkdir(parents=True, exist_ok=True)
Path("outputs/summary/synthetic_smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(manifest_path)
PY

COMMON_ARGS=(
  --config configs/debug.yaml
  --override data.cache_manifest_path=outputs/cache/synthetic_smoke_manifest.csv
  --override data.debug_max_records=32
  --override training.batch_size=8
  --override training.eval_batch_size=8
  --override training.num_workers=0
  --override training.epochs=1
)

"${PYTHON_BIN}" scripts/03_train_torch_model.py "${COMMON_ARGS[@]}" --model-name simple_cnn
"${PYTHON_BIN}" scripts/04_eval_model.py "${COMMON_ARGS[@]}" --model-name simple_cnn
"${PYTHON_BIN}" scripts/05_fit_calibration.py --config configs/debug.yaml --model-name simple_cnn
"${PYTHON_BIN}" scripts/06_eval_selective_policy.py --config configs/debug.yaml --model-name simple_cnn --calibration-method best
"${PYTHON_BIN}" scripts/07_eval_robustness.py "${COMMON_ARGS[@]}" --model-name simple_cnn
"${PYTHON_BIN}" scripts/08_generate_explanations.py --config configs/debug.yaml --model-name simple_cnn --calibration-method best --synthetic-smoke --override data.cache_manifest_path=outputs/cache/synthetic_smoke_manifest.csv --override explainability.num_samples=2
"${PYTHON_BIN}" scripts/09_build_figures.py --config configs/debug.yaml --model-name simple_cnn --calibration-method best --synthetic-smoke --override data.index_path=outputs/cache/synthetic_smoke_manifest.csv
"${PYTHON_BIN}" scripts/10_build_results_summary.py --config configs/debug.yaml --model-name simple_cnn --calibration-method best --synthetic-smoke

printf '%s\n' "Synthetic smoke summary: outputs/summary/synthetic_smoke_summary.json"
