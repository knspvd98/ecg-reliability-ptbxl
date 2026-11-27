PYTHON ?= python3
CONFIG ?= configs/base.yaml

.PHONY: install test lint prepare cache smoke suite workbench xgboost train eval calibrate selective robustness xai figures summary

install:
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	ruff check .

test:
	$(PYTHON) -m pytest

suite:
	ecg-reliability run-suite --config configs/debug.yaml --tier synthetic_smoke

workbench:
	ecg-reliability workbench --host 127.0.0.1 --port 8501

prepare:
	$(PYTHON) scripts/00_prepare_ptbxl.py --config $(CONFIG)

cache:
	$(PYTHON) scripts/01_cache_signals.py --config $(CONFIG)

smoke:
	bash runpod/smoke_run.sh

xgboost:
	$(PYTHON) scripts/02_train_xgboost.py --config configs/xgboost.yaml

train:
	$(PYTHON) scripts/03_train_torch_model.py --config $(CONFIG)

eval:
	$(PYTHON) scripts/04_eval_model.py --config $(CONFIG)

calibrate:
	$(PYTHON) scripts/05_fit_calibration.py --config $(CONFIG)

selective:
	$(PYTHON) scripts/06_eval_selective_policy.py --config $(CONFIG)

robustness:
	$(PYTHON) scripts/07_eval_robustness.py --config $(CONFIG)

xai:
	$(PYTHON) scripts/08_generate_explanations.py --config $(CONFIG)

figures:
	$(PYTHON) scripts/09_build_figures.py --config $(CONFIG)

summary:
	$(PYTHON) scripts/10_build_results_summary.py --config $(CONFIG)
