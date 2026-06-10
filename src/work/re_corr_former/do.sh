#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${REPO_ROOT}/.uv-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_ROOT}/.matplotlib-cache}"

OUTPUT_NAME="${OUTPUT_NAME:-ReCorrFormer}"
BACKEND="${BACKEND:-local}"
MAX_ASSETS="${MAX_ASSETS:-40}"
MIN_HISTORY_YEARS="${MIN_HISTORY_YEARS:-8.0}"
EPOCHS="${EPOCHS:-2}"
SHARPE_EPOCHS="${SHARPE_EPOCHS:-5}"
SAMPLE_STRIDE="${SAMPLE_STRIDE:-10}"
MAX_CANDIDATES_PER_DATE="${MAX_CANDIDATES_PER_DATE:-80}"
TOP_K="${TOP_K:-20}"
PAIR_DIAGNOSTIC_LIMIT="${PAIR_DIAGNOSTIC_LIMIT:-40}"
CANDIDATE_IMAGES="${CANDIDATE_IMAGES:-1}"
CANDIDATE_IMAGE_LIMIT="${CANDIDATE_IMAGE_LIMIT:-40}"

COMMON_ARGS=(
  --backend "${BACKEND}"
  --output-name "${OUTPUT_NAME}"
  --max-assets "${MAX_ASSETS}"
  --min-history-years "${MIN_HISTORY_YEARS}"
  --epochs "${EPOCHS}"
  --sharpe-epochs "${SHARPE_EPOCHS}"
  --sample-stride "${SAMPLE_STRIDE}"
  --max-candidates-per-date "${MAX_CANDIDATES_PER_DATE}"
  --top-k "${TOP_K}"
  --pair-diagnostic-limit "${PAIR_DIAGNOSTIC_LIMIT}"
  --candidate-preview-rows 0
)

if [[ "${CANDIDATE_IMAGES}" == "1" ]]; then
  COMMON_ARGS+=(--candidate-images --candidate-image-limit "${CANDIDATE_IMAGE_LIMIT}")
else
  COMMON_ARGS+=(--no-candidate-images)
fi

run_re_corr_former() {
  local trade_rule="$1"
  local selection_model="$2"
  local sharpe_model="${3:-lstm}"

  echo "[RUN] ReCorrFormer trade_rule=${trade_rule} selection_model=${selection_model} sharpe_model=${sharpe_model}"
  if [[ "${selection_model}" == "sharpe" ]]; then
    uv run src/work/re_corr_former/run.py \
      "${COMMON_ARGS[@]}" \
      --trade-rule "${trade_rule}" \
      --selection-model sharpe \
      --sharpe-model "${sharpe_model}"
  else
    uv run src/work/re_corr_former/run.py \
      "${COMMON_ARGS[@]}" \
      --trade-rule "${trade_rule}" \
      --selection-model corr
  fi
}

for trade_rule in vidyamurthy gatev; do
  run_re_corr_former "${trade_rule}" corr
  run_re_corr_former "${trade_rule}" sharpe lstm
  run_re_corr_former "${trade_rule}" sharpe transformer
done
