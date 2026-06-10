#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${REPO_ROOT}/.uv-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_ROOT}/.matplotlib-cache}"

OUTPUT_NAME="${OUTPUT_NAME:-ISEPT}"
MAX_ASSETS="${MAX_ASSETS:-20}"
MIN_HISTORY_YEARS="${MIN_HISTORY_YEARS:-8.0}"
MONTHS="${MONTHS:-8}"
CAE_EPOCHS="${CAE_EPOCHS:-20}"
MLP_EPOCHS="${MLP_EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-512}"
TOP_K_PAIRS="${TOP_K_PAIRS:-100}"
FEEDBACK_PAIRS_PER_SIDE="${FEEDBACK_PAIRS_PER_SIDE:-20}"
WARMUP_MONTHS="${WARMUP_MONTHS:-2}"

COMMON_ARGS=(
  --output-name "${OUTPUT_NAME}"
  --max-assets "${MAX_ASSETS}"
  --min-history-years "${MIN_HISTORY_YEARS}"
  --months "${MONTHS}"
  --cae-epochs "${CAE_EPOCHS}"
  --mlp-epochs "${MLP_EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --top-k-pairs "${TOP_K_PAIRS}"
  --feedback-pairs-per-side "${FEEDBACK_PAIRS_PER_SIDE}"
  --warmup-months "${WARMUP_MONTHS}"
)

run_isept() {
  local trade_rule="$1"
  echo "[RUN] ISEPT trade_rule=${trade_rule}"
  uv run "${SCRIPT_DIR}/run.py" "${COMMON_ARGS[@]}" --trade-rule "${trade_rule}"
}

for trade_rule in vidyamurthy gatev; do
  run_isept "${trade_rule}"
done
