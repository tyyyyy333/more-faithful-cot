#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LOCK_DIR=".v2_firstk_sweep_queue.lock"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "Another first-k sweep queue appears to be running: $LOCK_DIR" >&2
  exit 2
fi
trap 'rm -rf "$LOCK_DIR"' EXIT

RESULT_DIR="v2_outputs/final_results/arc-challenge/Phi-3"
SUMMARY_DIR="${SUMMARY_DIR:-v2_outputs/reproduction/firstk_sweep}"
K_VALUES="${K_VALUES:-1 2 8 16}"
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

running_for_prefix() {
  local prefix="$1"
  pgrep -af -- "--output_suffix ${prefix}_s" >/dev/null
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

  if running_for_prefix "$prefix"; then
    log "ERROR: $prefix is already running"
    exit 2
  fi

  local existing
  existing="$(final_count_for_prefix "$prefix")"
  if [[ "$existing" -ge "$NUM_PROCS" ]]; then
    log "Skipping $prefix; found existing complete result files ($existing/$NUM_PROCS)"
    summarize
    return 0
  fi
  if [[ "$existing" -gt 0 ]]; then
    log "ERROR: found partial $prefix result files ($existing/$NUM_PROCS). Move/delete them before rerunning."
    exit 2
  fi

  log "Starting $prefix"
  env "${COMMON_ENV[@]}" OUTPUT_PREFIX="$prefix" \
    bash v2/run_sharded_unlearn.sh "${COMMON_ARGS[@]}" \
      --forget_k_tokens "$k"

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    log "Dry-run complete for $prefix"
    return 0
  fi

  existing="$(final_count_for_prefix "$prefix")"
  if [[ "$existing" -lt "$NUM_PROCS" ]]; then
    log "ERROR: $prefix finished but only $existing/$NUM_PROCS final files exist"
    exit 1
  fi
  summarize
}

preflight() {
  log "Running preflight checks"
  test -f v2/run_sharded_unlearn.sh
  test -f v2/summarize_results.py
  test -f v2/summarize_mechanistic_diagnostics.py
  test -d "$RESULT_DIR"
  log "Using K_VALUES=$K_VALUES"
  log "Using NUM_PROCS=$NUM_PROCS GPU_IDS=$GPU_IDS CONDA_ENV=$CONDA_ENV"
  log "Preflight checks passed"
}

preflight

COMMON_ENV=(
  NUM_PROCS="$NUM_PROCS"
  GPU_IDS="$GPU_IDS"
  CONDA_ENV="$CONDA_ENV"
  BATCH_SIZE=8
  ADAPTER_GROUP_SIZE=8
  EVAL_INTERVAL=5
  LORA_RANK=16
  LORA_ALPHA=32
  LR=5e-4
)

COMMON_ARGS=(
  --mechanistic_diag
)

for k in $K_VALUES; do
  run_k "$k"
done

log "First-k sweep queue complete"
