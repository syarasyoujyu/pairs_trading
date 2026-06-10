#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${REPO_ROOT}/.uv-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_ROOT}/.matplotlib-cache}"

OUTPUT_NAME="${OUTPUT_NAME:-EnhancingPairsTradingML}"
MAX_SELECTED_PAIRS="${MAX_SELECTED_PAIRS:-20}"
NEURAL_BACKEND="${NEURAL_BACKEND:-modal}"
MODAL_SAVE_MODELS="${MODAL_SAVE_MODELS:-1}"

COMMON_ARGS=(
  --output-name "${OUTPUT_NAME}"
  --max-selected-pairs "${MAX_SELECTED_PAIRS}"
)

MODAL_SAVE_ARGS=(--modal-save-models)
if [[ "${MODAL_SAVE_MODELS}" != "1" ]]; then
  MODAL_SAVE_ARGS=(--no-modal-save-models)
fi

run_enhancing() {
  local forecast_model="$1"
  echo "[RUN] EnhancingPairsTradingML forecast_model=${forecast_model}"
  if [[ "${forecast_model}" == "lstm" || "${forecast_model}" == "encoder_decoder" ]]; then
    uv run "${SCRIPT_DIR}/run.py" \
      "${COMMON_ARGS[@]}" \
      --forecast-model "${forecast_model}" \
      --neural-backend "${NEURAL_BACKEND}" \
      "${MODAL_SAVE_ARGS[@]}"
  else
    uv run "${SCRIPT_DIR}/run.py" \
      "${COMMON_ARGS[@]}" \
      --forecast-model "${forecast_model}"
  fi
}

for forecast_model in arma rolling_ar lstm encoder_decoder; do
  run_enhancing "${forecast_model}"
done
