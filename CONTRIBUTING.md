# Contributing

## Local checks

Create a virtual environment and install the development dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Before opening a pull request, run:

```bash
ruff check .
pytest
bash runpod/smoke_run.sh
```

## Research changes

- Keep train, validation, and test folds isolated.
- Fit calibration, thresholds, and selection policies on validation data only.
- Add or update tests for metrics, data transformations, and claim artifacts.
- Do not commit datasets, generated signals, checkpoints, predictions, or secrets.
- Label synthetic smoke results clearly; never present them as scientific findings.
