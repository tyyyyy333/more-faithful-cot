#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-pf-a100}"
NUM_PROCS="${NUM_PROCS:-2}"
GPU_IDS="${GPU_IDS:-0}"
LOG_DIR="${LOG_DIR:-logs}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-strong_${NUM_PROCS}proc}"
DRY_RUN="${DRY_RUN:-0}"

MODEL_NAME="${MODEL_NAME:-microsoft/Phi-3-mini-4k-instruct}"
DATASET="${DATASET:-arc-challenge}"
STRATEGY="${STRATEGY:-sentencize}"
METHOD="${METHOD:-npo_KL}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-1e-4}"
COT_LIMIT="${COT_LIMIT:-250}"
VERIFY_SIZE="${VERIFY_SIZE:-20}"
RETAIN_N="${RETAIN_N:-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EVAL_INTERVAL="${EVAL_INTERVAL:-5}"
ADAPTER_GROUP_SIZE="${ADAPTER_GROUP_SIZE:-8}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
POS="${POS:-1}"
SKIP_INITIAL_EVAL="${SKIP_INITIAL_EVAL:-0}"
SKIP_SPECIFICITY="${SKIP_SPECIFICITY:-0}"
SKIP_NEW_COT="${SKIP_NEW_COT:-0}"

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
if [[ "$NUM_PROCS" -lt 1 ]]; then
  echo "NUM_PROCS must be >= 1" >&2
  exit 2
fi
if [[ "${#GPU_ARRAY[@]}" -lt 1 ]]; then
  echo "GPU_IDS must contain at least one GPU id" >&2
  exit 2
fi

mkdir -p "$LOG_DIR"

common_args=(
  v2/unlearn.py
  --model_name "$MODEL_NAME"
  --dataset "$DATASET"
  --strategy "$STRATEGY"
  --stepwise
  --method "$METHOD"
  --epochs "$EPOCHS"
  --lr "$LR"
  --cot_limit "$COT_LIMIT"
  --verify_size "$VERIFY_SIZE"
  --retain_n "$RETAIN_N"
  --batch_size "$BATCH_SIZE"
  --eval_interval "$EVAL_INTERVAL"
  --device cuda
  --device_map none
  --adapter_group_size "$ADAPTER_GROUP_SIZE"
  --lora_rank "$LORA_RANK"
  --lora_alpha "$LORA_ALPHA"
)

if [[ "$POS" == "1" ]]; then
  common_args+=(--pos)
fi
if [[ "$SKIP_INITIAL_EVAL" == "1" ]]; then
  common_args+=(--skip_initial_eval)
fi
if [[ "$SKIP_SPECIFICITY" == "1" ]]; then
  common_args+=(--skip_specificity)
fi
if [[ "$SKIP_NEW_COT" == "1" ]]; then
  common_args+=(--skip_new_cot)
fi

pids=()
for ((shard=0; shard<NUM_PROCS; shard++)); do
  gpu="${GPU_ARRAY[$((shard % ${#GPU_ARRAY[@]}))]}"
  suffix="${OUTPUT_PREFIX}_s${shard}"
  log_file="${LOG_DIR}/v2_${suffix}.log"
  cmd=(
    conda run -n "$CONDA_ENV" python
    "${common_args[@]}"
    --job_shard_count "$NUM_PROCS"
    --job_shard_index "$shard"
    --output_suffix "$suffix"
    "$@"
  )

  echo "Launching shard ${shard}/${NUM_PROCS} on GPU ${gpu}; log: ${log_file}"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$gpu"
    printf '%q ' "${cmd[@]}"
    printf '> %q 2>&1\n' "$log_file"
    continue
  fi

  CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" > "$log_file" 2>&1 &
  pids+=("$!")
done

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

exit "$status"
