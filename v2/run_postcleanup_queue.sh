#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LOCK_DIR=".v2_postcleanup_queue.lock"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  if ! pgrep -af "run_postcleanup_queue.sh" >/dev/null; then
    rmdir "$LOCK_DIR" 2>/dev/null || true
    mkdir "$LOCK_DIR"
  else
    echo "Another post-cleanup queue appears to be running: $LOCK_DIR" >&2
    exit 2
  fi
fi
trap 'rm -rf "$LOCK_DIR"' EXIT

RESULT_DIR="v2_outputs/final_results/arc-challenge/Phi-3"
SUMMARY_DIR="v2_outputs/reproduction/summary"
NUM_PROCS="${NUM_PROCS:-2}"
GPU_IDS="${GPU_IDS:-0}"
CONDA_ENV="${CONDA_ENV:-pf-a100}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

timestamp() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

running_for_prefix() {
  local prefix="$1"
  pgrep -af -- "--output_suffix ${prefix}_s" >/dev/null
}

final_count_for_prefix() {
  local prefix="$1"
  find "$RESULT_DIR" -maxdepth 1 -type f -name "*_${prefix}_s*.out" | wc -l
}

summarize() {
  log "Refreshing summary from active final_results"
  conda run -n "$CONDA_ENV" python v2/summarize_results.py \
    --results-root v2_outputs/final_results \
    --outdir "$SUMMARY_DIR"
}

wait_for_existing_or_run() {
  local prefix="$1"
  shift

  local existing
  existing="$(final_count_for_prefix "$prefix")"
  if [[ "$existing" -ge "$NUM_PROCS" ]]; then
    log "Skipping $prefix: found $existing final result files"
    summarize
    return 0
  fi

  if running_for_prefix "$prefix"; then
    log "Waiting for already-running experiment: $prefix"
    while running_for_prefix "$prefix"; do
      sleep 300
      log "Still waiting for $prefix"
    done
    existing="$(final_count_for_prefix "$prefix")"
    if [[ "$existing" -lt "$NUM_PROCS" ]]; then
      log "ERROR: $prefix stopped but only $existing/$NUM_PROCS final files exist"
      exit 1
    fi
    summarize
    return 0
  fi

  log "Starting experiment: $prefix"
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
  test -x /bin/bash
  test -f v2/run_sharded_unlearn.sh
  test -f v2/summarize_results.py
  test -d "$RESULT_DIR"
  log "Preflight checks passed"
}

preflight

wait_for_existing_or_run \
  "supp_phi3_arc_alllinear_lr3e4_postcleanup" \
  env NUM_PROCS="$NUM_PROCS" GPU_IDS="$GPU_IDS" \
    OUTPUT_PREFIX="supp_phi3_arc_alllinear_lr3e4_postcleanup" \
    BATCH_SIZE=1 ADAPTER_GROUP_SIZE=1 EVAL_INTERVAL=1 \
    LORA_RANK=16 LORA_ALPHA=32 LR=3e-4 \
    bash v2/run_sharded_unlearn.sh

wait_for_existing_or_run \
  "supp_phi3_arc_alllinear_r32_lr5e4_postcleanup" \
  env NUM_PROCS="$NUM_PROCS" GPU_IDS="$GPU_IDS" \
    OUTPUT_PREFIX="supp_phi3_arc_alllinear_r32_lr5e4_postcleanup" \
    BATCH_SIZE=1 ADAPTER_GROUP_SIZE=1 EVAL_INTERVAL=1 \
    LORA_RANK=32 LORA_ALPHA=32 LR=5e-4 \
    bash v2/run_sharded_unlearn.sh

log "Post-cleanup experiment queue complete"
