#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate 2>/dev/null || true

python scripts/00_prepare_ptbxl.py --config configs/debug.yaml
python scripts/01_cache_signals.py --config configs/debug.yaml
python scripts/03_train_torch_model.py --config configs/debug.yaml --model-name simple_cnn
python scripts/04_eval_model.py --config configs/debug.yaml --model-name simple_cnn
python scripts/05_fit_calibration.py --config configs/debug.yaml --model-name simple_cnn
python scripts/06_eval_selective_policy.py --config configs/debug.yaml --model-name simple_cnn --calibration-method best
python scripts/07_eval_robustness.py --config configs/debug.yaml --model-name simple_cnn
python scripts/08_generate_explanations.py --config configs/debug.yaml --model-name simple_cnn --calibration-method best
python scripts/09_build_figures.py --config configs/debug.yaml --model-name simple_cnn --calibration-method best
python scripts/10_build_results_summary.py --config configs/debug.yaml --model-name simple_cnn --calibration-method best

printf '%s\n' "outputs/index/index_summary.json"
printf '%s\n' "outputs/cache/cache_summary.json"
printf '%s\n' "outputs/summary/results_summary.json"
