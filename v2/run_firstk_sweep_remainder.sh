#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RESULT_DIR="v2_outputs/final_results/arc-challenge/Phi-3"
SUMMARY_DIR="${SUMMARY_DIR:-v2_outputs/reproduction/firstk_sweep}"
K_VALUES="${K_VALUES:-2 8 16}"
NUM_PROCS="${NUM_PROCS:-2}"
GPU_IDS="${GPU_IDS:-0}"
CONDA_ENV="${CONDA_ENV:-pf-a100}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

timestamp() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

final_count_for_prefix() {
  local prefix="$1"
  find "$RESULT_DIR" -maxdepth 1 -type f -name "*_${prefix}_s*.out" | wc -l
}

line_count_for_prefix() {
  local prefix="$1"
  find "$RESULT_DIR" -maxdepth 1 -type f -name "*_${prefix}_s*.out" -print0 \
    | xargs -0 -r wc -l \
    | awk '/ total$/ {print $1} END {if (NR == 1) print $1; if (NR == 0) print 0}'
}

running_firstk() {
  pgrep -af 'test_firstk[0-9]+_only_diag|run_sharded_unlearn.sh --mechanistic_diag --forget_k_tokens' >/dev/null
}

summarize() {
  mkdir -p "$SUMMARY_DIR"
  conda run -n "$CONDA_ENV" python v2/summarize_results.py \
    --results-root v2_outputs/final_results \
    --pattern "*test_firstk*_only_diag_s*.out" \
    --outdir "$SUMMARY_DIR/summary"
  conda run -n "$CONDA_ENV" python v2/summarize_mechanistic_diagnostics.py \
    --results-dir "$RESULT_DIR" \
    --pattern "*test_firstk*_only_diag_s*.out" \
    --outdir "$SUMMARY_DIR/mechanistic"
}

run_k() {
  local k="$1"
  local prefix="test_firstk${k}_only_diag"
  local existing
  existing="$(final_count_for_prefix "$prefix")"

  if [[ "$existing" -ge "$NUM_PROCS" ]]; then
    log "Skipping $prefix; found $existing result files"
    return 0
  fi

  log "Starting $prefix with conservative batch/group size"
  NUM_PROCS="$NUM_PROCS" GPU_IDS="$GPU_IDS" CONDA_ENV="$CONDA_ENV" \
    OUTPUT_PREFIX="$prefix" BATCH_SIZE="${BATCH_SIZE:-4}" ADAPTER_GROUP_SIZE="${ADAPTER_GROUP_SIZE:-4}" \
    EVAL_INTERVAL=5 LORA_RANK=16 LORA_ALPHA=32 LR=5e-4 \
    bash v2/run_sharded_unlearn.sh --mechanistic_diag --forget_k_tokens "$k"
  summarize
}

log "Waiting for any active first-k run to finish"
while running_firstk; do
  sleep 60
done

log "Initial line counts after wait:"
for k in 1 4; do
  log "test_firstk${k}_only_diag rows=$(line_count_for_prefix "test_firstk${k}_only_diag")"
done

for k in $K_VALUES; do
  run_k "$k"
done

summarize
log "First-k remainder queue complete"
