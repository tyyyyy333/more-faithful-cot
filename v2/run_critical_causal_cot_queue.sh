#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-pf-a100}"
NUM_PROCS="${NUM_PROCS:-2}"
GPU_IDS="${GPU_IDS:-0}"
LOG_DIR="${LOG_DIR:-logs}"
RESULTS_ROOT="${RESULTS_ROOT:-v2_outputs/final_results}"
REPRO_ROOT="${REPRO_ROOT:-v2_outputs/reproduction/critical_lora}"
DRY_RUN="${DRY_RUN:-0}"
IMPORTANCE_FILE="${IMPORTANCE_FILE:-v2_outputs/step_importance/arc-challenge_Phi-3_step_importance_full.csv}"
MIN_IMPORTANCE_SCORE="${MIN_IMPORTANCE_SCORE:-0.9}"
MAX_IMPORTANCE_STEPS="${MAX_IMPORTANCE_STEPS:-0}"

MODEL_NAME="${MODEL_NAME:-microsoft/Phi-3-mini-4k-instruct}"
DATASET="${DATASET:-arc-challenge}"
STRATEGY="${STRATEGY:-sentencize}"
METHOD="${METHOD:-npo_KL}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-5e-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
ADAPTER_GROUP_SIZE="${ADAPTER_GROUP_SIZE:-1}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-32}"
POS="${POS:-1}"
CAUSAL_COT_LAMBDA="${CAUSAL_COT_LAMBDA:-0.1}"
CAUSAL_COT_MARGIN="${CAUSAL_COT_MARGIN:-1.0}"
CAUSAL_COT_COUNTERFACTUAL="${CAUSAL_COT_COUNTERFACTUAL:-remove_step}"
CAUSAL_COT_ANSWER="${CAUSAL_COT_ANSWER:-correct}"

OUTPUT_PREFIX="${OUTPUT_PREFIX:-critical_lora_causal_cot_i09}"

log() {
  printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"
}

if [[ ! -f "$IMPORTANCE_FILE" ]]; then
  echo "Missing IMPORTANCE_FILE: $IMPORTANCE_FILE" >&2
  exit 2
fi

mkdir -p "$LOG_DIR" "$REPRO_ROOT"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

if compgen -G "$RESULTS_ROOT/$DATASET/Phi-3/*_${OUTPUT_PREFIX}_s*.out" > /dev/null; then
  echo "Existing result files found for ${OUTPUT_PREFIX}; choose a new OUTPUT_PREFIX or clean stale files." >&2
  exit 2
fi

log "Starting ${OUTPUT_PREFIX}"
CONDA_ENV="$CONDA_ENV" NUM_PROCS="$NUM_PROCS" GPU_IDS="$GPU_IDS" OUTPUT_PREFIX="$OUTPUT_PREFIX" \
MODEL_NAME="$MODEL_NAME" DATASET="$DATASET" STRATEGY="$STRATEGY" METHOD="$METHOD" \
EPOCHS="$EPOCHS" LR="$LR" BATCH_SIZE="$BATCH_SIZE" ADAPTER_GROUP_SIZE="$ADAPTER_GROUP_SIZE" \
LORA_RANK="$LORA_RANK" LORA_ALPHA="$LORA_ALPHA" POS="$POS" \
bash v2/run_sharded_unlearn.sh \
  --mechanistic_diag \
  --step_importance_file "$IMPORTANCE_FILE" \
  --min_importance_score "$MIN_IMPORTANCE_SCORE" \
  --max_importance_steps "$MAX_IMPORTANCE_STEPS" \
  --causal_cot_loss \
  --causal_cot_lambda "$CAUSAL_COT_LAMBDA" \
  --causal_cot_margin "$CAUSAL_COT_MARGIN" \
  --causal_cot_counterfactual "$CAUSAL_COT_COUNTERFACTUAL" \
  --causal_cot_answer "$CAUSAL_COT_ANSWER" \
  --causal_cot_auto_scale

if [[ "$DRY_RUN" == "1" ]]; then
  log "Dry run complete"
  exit 0
fi

conda run -n "$CONDA_ENV" python v2/summarize_results.py \
  --results-root "$RESULTS_ROOT" \
  --outdir "$REPRO_ROOT/summary"

conda run -n "$CONDA_ENV" python v2/summarize_mechanistic_diagnostics.py \
  --results-dir "$RESULTS_ROOT/$DATASET/Phi-3" \
  --pattern "*${OUTPUT_PREFIX}*.out" \
  --outdir "$REPRO_ROOT/mechanistic"

log "Critical Causal_CoT queue complete"
