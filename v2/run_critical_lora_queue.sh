#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LOCK_DIR=".v2_critical_lora_queue.lock"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "Another critical LoRA queue appears to be running: $LOCK_DIR" >&2
  exit 2
fi
trap 'rm -rf "$LOCK_DIR"' EXIT

RESULT_DIR="v2_outputs/final_results/arc-challenge/Phi-3"
SUMMARY_DIR="v2_outputs/reproduction/critical_lora"
IMPORTANCE_FILE="${IMPORTANCE_FILE:-v2_outputs/step_importance/arc-challenge_Phi-3_step_importance_full.csv}"
MIN_IMPORTANCE_SCORE="${MIN_IMPORTANCE_SCORE:-0.9}"
MAX_IMPORTANCE_STEPS="${MAX_IMPORTANCE_STEPS:-0}"
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
    --outdir "$SUMMARY_DIR/summary"
  conda run -n "$CONDA_ENV" python v2/summarize_mechanistic_diagnostics.py \
    --results-dir "$RESULT_DIR" \
    --pattern "*critical_lora_baseline_i09_s*.out" \
    --pattern "*critical_lora_repr_i09_s*.out" \
    --outdir "$SUMMARY_DIR/mechanistic"
}

run_once() {
  local prefix="$1"
  shift

  if running_for_prefix "$prefix"; then
    log "ERROR: $prefix is already running"
    exit 2
  fi

  local existing
  existing="$(final_count_for_prefix "$prefix")"
  if [[ "$existing" -gt 0 ]]; then
    log "ERROR: found existing $prefix result files ($existing). Move/delete them before rerunning."
    exit 2
  fi

  log "Starting $prefix"
  "$@"

  existing="$(final_count_for_prefix "$prefix")"
  if [[ "$existing" -lt "$NUM_PROCS" ]]; then
    log "ERROR: $prefix finished but only $existing/$NUM_PROCS final files exist"
    exit 1
  fi
  summarize
}

preflight() {
  log "Running preflight checks"
  test -f "$IMPORTANCE_FILE"
  test -f v2/run_sharded_unlearn.sh
  test -f v2/summarize_results.py
  test -f v2/summarize_mechanistic_diagnostics.py
  test -d "$RESULT_DIR"
  log "Using IMPORTANCE_FILE=$IMPORTANCE_FILE"
  log "Using MIN_IMPORTANCE_SCORE=$MIN_IMPORTANCE_SCORE"
  log "Using MAX_IMPORTANCE_STEPS=$MAX_IMPORTANCE_STEPS"
  log "Preflight checks passed"
}

preflight

COMMON_ENV=(
  NUM_PROCS="$NUM_PROCS"
  GPU_IDS="$GPU_IDS"
  CONDA_ENV="$CONDA_ENV"
  BATCH_SIZE=1
  ADAPTER_GROUP_SIZE=1
  EVAL_INTERVAL=1
  LORA_RANK=32
  LORA_ALPHA=32
  LR=5e-4
)

COMMON_ARGS=(
  --mechanistic_diag
  --step_importance_file "$IMPORTANCE_FILE"
  --min_importance_score "$MIN_IMPORTANCE_SCORE"
  --max_importance_steps "$MAX_IMPORTANCE_STEPS"
)

run_once \
  "critical_lora_baseline_i09" \
  env "${COMMON_ENV[@]}" OUTPUT_PREFIX="critical_lora_baseline_i09" \
    bash v2/run_sharded_unlearn.sh "${COMMON_ARGS[@]}"

run_once \
  "critical_lora_repr_i09" \
  env "${COMMON_ENV[@]}" OUTPUT_PREFIX="critical_lora_repr_i09" \
    bash v2/run_sharded_unlearn.sh "${COMMON_ARGS[@]}" \
      --repr_loss \
      --repr_lambda 0.1 \
      --repr_last_layers 4 \
      --repr_gamma 0.9 \
      --repr_k_tokens 4 \
      --repr_auto_scale

log "Critical LoRA queue complete"
