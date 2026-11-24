#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate 2>/dev/null || true

export CLEAN_OUTPUTS="${CLEAN_OUTPUTS:-1}"
export RESUME="${RESUME:-0}"
export INCLUDE_TRANSFORMER="${INCLUDE_TRANSFORMER:-0}"
export INCLUDE_TRANSFORMER_IN_SELECTION="${INCLUDE_TRANSFORMER_IN_SELECTION:-0}"
export INCLUDE_TRANSFORMER_REQUIRED="${INCLUDE_TRANSFORMER_REQUIRED:-0}"
export RUN_BOOTSTRAP="${RUN_BOOTSTRAP:-1}"
export N_BOOTSTRAP="${N_BOOTSTRAP:-500}"
export INCLUDE_EXTERNAL="${INCLUDE_EXTERNAL:-1}"
export EXTERNAL_SOURCES="${EXTERNAL_SOURCES:-georgia,chapman_shaoxing}"

if [[ -z "${PTBXL_ROOT:-}" ]]; then
  echo "PTBXL_ROOT is required."
  exit 1
fi

if [[ "${INCLUDE_EXTERNAL}" == "1" ]]; then
  missing_direct=0
  IFS=',' read -r -a external_sources_array <<< "${EXTERNAL_SOURCES}"
  for source_name in "${external_sources_array[@]}"; do
    case "${source_name}" in
      georgia)
        [[ -n "${GEORGIA_ROOT:-}" ]] || missing_direct=1
        ;;
      chapman_shaoxing)
        [[ -n "${CHAPMAN_SHAOXING_ROOT:-}" ]] || missing_direct=1
        ;;
      ningbo)
        [[ -n "${NINGBO_ROOT:-}" ]] || missing_direct=1
        ;;
    esac
  done
  if [[ "${missing_direct}" == "1" && -z "${CHALLENGE2021_ROOT:-}" ]]; then
    echo "External validation requires direct source roots (GEORGIA_ROOT / CHAPMAN_SHAOXING_ROOT / NINGBO_ROOT) or optional CHALLENGE2021_ROOT fallback."
    exit 1
  fi
fi

if [[ "${CLEAN_OUTPUTS}" == "1" && "${RESUME}" != "1" ]]; then
  rm -rf outputs
fi
mkdir -p outputs/campaign/logs

python - <<'PY'
import os
from ecg_reliability.campaign.manifest import build_campaign_manifest, write_campaign_manifest

payload = build_campaign_manifest(
    ptbxl_root=os.environ.get("PTBXL_ROOT"),
    challenge2021_root=os.environ.get("CHALLENGE2021_ROOT"),
    clean_outputs=os.environ.get("CLEAN_OUTPUTS"),
    resume=os.environ.get("RESUME"),
    include_transformer=os.environ.get("INCLUDE_TRANSFORMER"),
    include_transformer_in_selection=os.environ.get("INCLUDE_TRANSFORMER_IN_SELECTION"),
    include_external=os.environ.get("INCLUDE_EXTERNAL"),
    external_sources=os.environ.get("EXTERNAL_SOURCES"),
    n_bootstrap=os.environ.get("N_BOOTSTRAP"),
)
write_campaign_manifest("outputs/campaign/manifest.json", payload)
PY

run_stage() {
  local stage_name="$1"
  local command="$2"
  shift 2 || true
  local extra_args=()
  if [[ "${RESUME}" == "1" ]]; then
    extra_args+=(--resume)
  fi
  python -m ecg_reliability.campaign.stage_runner --stage-name "${stage_name}" --command "${command}" "${extra_args[@]}" "$@"
}

skip_stage() {
  local stage_name="$1"
  local reason="$2"
  local extra_args=()
  if [[ "${RESUME}" == "1" ]]; then
    extra_args+=(--resume)
  fi
  python -m ecg_reliability.campaign.stage_runner --stage-name "${stage_name}" --skip --skip-reason "${reason}" "${extra_args[@]}"
}

run_stage "stage_01_real_ptbxl_smoke" "bash runpod/smoke_real_ptbxl.sh"
run_stage "stage_02_performance_smoke" "bash runpod/performance_smoke.sh"
run_stage "stage_03_xgboost" "python scripts/02_train_xgboost.py --config configs/xgboost.yaml && python scripts/04_eval_model.py --config configs/xgboost.yaml --model-name xgboost"
run_stage "stage_04_resnet1d" "python scripts/03_train_torch_model.py --config configs/resnet1d.yaml --model-name resnet1d && python scripts/04_eval_model.py --config configs/resnet1d.yaml --model-name resnet1d"
run_stage "stage_05_inceptiontime" "python scripts/03_train_torch_model.py --config configs/inceptiontime.yaml --model-name inceptiontime && python scripts/04_eval_model.py --config configs/inceptiontime.yaml --model-name inceptiontime"
run_stage "stage_06_tcn" "python scripts/03_train_torch_model.py --config configs/tcn.yaml --model-name tcn && python scripts/04_eval_model.py --config configs/tcn.yaml --model-name tcn"

if [[ "${INCLUDE_TRANSFORMER}" == "1" ]]; then
  transformer_args=()
  if [[ "${INCLUDE_TRANSFORMER_REQUIRED}" != "1" ]]; then
    transformer_args+=(--allow-failure)
  fi
  run_stage "stage_07_transformer_optional" "python scripts/11_runtime_preflight.py --config configs/transformer.yaml --model-name transformer --max-batches 10 && python scripts/03_train_torch_model.py --config configs/transformer.yaml --model-name transformer && python scripts/04_eval_model.py --config configs/transformer.yaml --model-name transformer" "${transformer_args[@]}"
else
  skip_stage "stage_07_transformer_optional" "transformer_disabled"
fi

run_stage "stage_08_calibration_and_selective" "python - <<'PY'
from pathlib import Path
import subprocess
models = sorted({path.parts[-3] for path in Path('outputs/predictions').glob('*/raw/test_metrics_raw.json')})
for model in models:
    subprocess.run(['python', 'scripts/05_fit_calibration.py', '--config', 'configs/base.yaml', '--model-name', model], check=True)
    subprocess.run(['python', 'scripts/06_eval_selective_policy.py', '--config', 'configs/base.yaml', '--model-name', model, '--calibration-method', 'best'], check=True)
PY"

SELECTION_CANDIDATES="resnet1d inceptiontime tcn"
if [[ "${INCLUDE_TRANSFORMER}" == "1" && "${INCLUDE_TRANSFORMER_IN_SELECTION}" == "1" ]]; then
  SELECTION_CANDIDATES="${SELECTION_CANDIDATES} transformer"
fi
run_stage "stage_09_select_deep_model" "python scripts/13_select_deep_model.py --config configs/base.yaml --candidates ${SELECTION_CANDIDATES}"

SELECTED_DEEP_MODEL="$(python - <<'PY'
import json
from pathlib import Path
payload = json.loads(Path('outputs/summary/model_selection.json').read_text(encoding='utf-8'))
print(payload['selected_deep_model'])
PY
)"
export SELECTED_DEEP_MODEL

run_stage "stage_10_selected_model_robustness_xai" "python scripts/05_fit_calibration.py --config configs/base.yaml --model-name ${SELECTED_DEEP_MODEL} && python scripts/06_eval_selective_policy.py --config configs/base.yaml --model-name ${SELECTED_DEEP_MODEL} --calibration-method best && python scripts/07_eval_robustness.py --config configs/base.yaml --model-name ${SELECTED_DEEP_MODEL} --calibration-method best && python scripts/08_generate_explanations.py --config configs/base.yaml --model-name ${SELECTED_DEEP_MODEL} --calibration-method best --selection-manifest outputs/summary/model_selection.json"

if [[ "${INCLUDE_EXTERNAL}" == "1" ]]; then
  EXTERNAL_SOURCE_OVERRIDE="[${EXTERNAL_SOURCES}]"
  run_stage "stage_11_external_validation" "python scripts/18_prepare_external_challenge2021.py --config configs/base.yaml --override external.enabled=true --override external.sources=${EXTERNAL_SOURCE_OVERRIDE} && python scripts/19_eval_external.py --config configs/base.yaml --model-name ${SELECTED_DEEP_MODEL} --calibration-method best --override external.enabled=true --override external.sources=${EXTERNAL_SOURCE_OVERRIDE}"
else
  skip_stage "stage_11_external_validation" "external_disabled"
fi

run_stage "stage_12_figures_summary_tables" "python scripts/09_build_figures.py --config configs/base.yaml --model-name ${SELECTED_DEEP_MODEL} --calibration-method best --selection-manifest outputs/summary/model_selection.json && python scripts/10_build_results_summary.py --config configs/base.yaml --model-name ${SELECTED_DEEP_MODEL} --selection-manifest outputs/summary/model_selection.json && python scripts/15_build_paper_tables.py --config configs/base.yaml --selection-manifest outputs/summary/model_selection.json"

if [[ "${RUN_BOOTSTRAP}" == "1" ]]; then
  run_stage "stage_13_bootstrap" "python scripts/16_bootstrap_claim_metrics.py --config configs/base.yaml --n-bootstrap ${N_BOOTSTRAP}"
else
  skip_stage "stage_13_bootstrap" "bootstrap_disabled"
fi

run_stage "stage_14_validate_artifacts" "python scripts/14_validate_claim_artifacts.py --config configs/base.yaml --selection-manifest outputs/summary/model_selection.json"

run_stage "stage_15_archive" "python - <<'PY'
import tarfile
from pathlib import Path
items = [
    'outputs/index',
    'outputs/cache/cache_summary.json',
    'outputs/checkpoints',
    'outputs/predictions',
    'outputs/calibration',
    'outputs/decision',
    'outputs/robustness',
    'outputs/explanations',
    'outputs/external',
    'outputs/figures',
    'outputs/tables',
    'outputs/summary',
    'outputs/campaign',
]
archive_path = Path('outputs/ptbxl_claim_run_artifacts.tar.gz')
archive_path.parent.mkdir(parents=True, exist_ok=True)
with tarfile.open(archive_path, 'w:gz') as tar:
    for item in items:
        path = Path(item)
        if path.exists():
            tar.add(path, arcname=str(path))
print(archive_path)
PY"

printf '%s\n' "outputs/summary/results_summary.json"
printf '%s\n' "outputs/summary/model_selection.json"
printf '%s\n' "outputs/summary/claim_artifact_validation.json"
printf '%s\n' "outputs/tables/"
printf '%s\n' "outputs/figures/"
printf '%s\n' "outputs/external/"
printf '%s\n' "outputs/ptbxl_claim_run_artifacts.tar.gz"
