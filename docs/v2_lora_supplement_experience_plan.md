# v2 LoRA-FUR Supplement Experience Plan

## Summary

This plan treats `v2` as a LoRA-based supplement to the paper, not as an exact reproduction of the paper's full/FF2 main experiments.

The current implementation is sufficient for this supplement:

- Stepwise jobs are supported: each CoT step maps to one adapter job.
- Multi-process sharding is supported through `NUM_PROCS`, `--job_shard_count`, and `--job_shard_index`.
- Paper-style content-word filtering is supported through `--pos`; `v2/run_sharded_unlearn.sh` now defaults to `POS=1`.
- LoRA can target only `down_proj` through `--lora_target_modules down_proj`, matching the paper's LoRA appendix setup more closely than the all-linear default.
- Epoch 0, final, specificity, and new-CoT evaluation are supported.
- `v2/summarize_results.py` reports `ff_hard_pct`, `ff_soft_pct_mean`, `efficacy_mean`, `specificity_mean`, and loss summaries.

Paper reference for the closest main experiment:

| Paper item | Setting / metric |
| --- | --- |
| Model | `Phi-3-mini-4k-Instruct` |
| Dataset | `ARC-Challenge` |
| Method | `NPO+KL`, stepwise, content-word targets |
| Instances | 250 sampled test instances |
| Iterations | 5 |
| Retain examples | 4 same-dataset CoT steps |
| Main-result Eff | `40.8` |
| Main-result Spec | `99.5` |
| Main-result FUR / FF-HARD | `39.1` |

The paper's main metrics above are full/FF2 results. The commands below test whether v2 LoRA-FUR can reproduce the same trend.

## Experiment Matrix

All experiments use the same process shape requested for v2 testing:

```bash
NUM_PROCS=2 GPU_IDS=0 BATCH_SIZE=1 ADAPTER_GROUP_SIZE=1 EVAL_INTERVAL=1 bash v2/run_sharded_unlearn.sh
```

Use unique `OUTPUT_PREFIX` values. Do not run multiple experiments with the same `OUTPUT_PREFIX`, because v2 resumes/skips by output file.

### 1. Paper-Mapped LoRA-DownProj Anchor

Parameters:

| Parameter | Value |
| --- | --- |
| `MODEL_NAME` | `microsoft/Phi-3-mini-4k-instruct` |
| `DATASET` | `arc-challenge` |
| `STRATEGY` | `sentencize` |
| `METHOD` | `npo_KL` |
| `EPOCHS` | `5` |
| `LR` | `1e-4` |
| `COT_LIMIT` | `250` |
| `VERIFY_SIZE` | `20` |
| `RETAIN_N` | `4` |
| `POS` | `1` |
| `BATCH_SIZE` | `1` |
| `ADAPTER_GROUP_SIZE` | `1` |
| `LORA_RANK` | `16` |
| `LORA_ALPHA` | `32` |
| `lora_target_modules` | `down_proj` |

Paper comparison:

- Aligns with the paper's Phi-3 + ARC-Challenge main entry in data protocol, method, iterations, retain count, and POS filtering.
- Differs from the paper main entry because v2 uses LoRA adapters instead of full/FF2 parameter updates.
- Paper reference: `Eff=40.8`, `Spec=99.5`, `FUR/FF-HARD=39.1`.

Reason:

- This is the cleanest v2 LoRA approximation of the paper main setting.
- `down_proj` avoids modifying attention and follows the paper LoRA appendix target module.

Command:

```bash
NUM_PROCS=2 GPU_IDS=0 OUTPUT_PREFIX=supp_phi3_arc_downproj_lr1e4 BATCH_SIZE=1 ADAPTER_GROUP_SIZE=1 EVAL_INTERVAL=1 LORA_RANK=16 LORA_ALPHA=32 LR=1e-4 bash v2/run_sharded_unlearn.sh --lora_target_modules down_proj
```

### 2. Stronger LoRA-DownProj Calibration

Parameters:

| Parameter | Value |
| --- | --- |
| Base setting | Same as experiment 1 |
| `LR` | `3e-4` |

Paper comparison:

- Paper Table 8 small-sample Phi-3 + ARC-Challenge LR sweep reports `lr=1e-4: Eff=34.4, Spec=99.4, FF=53.3` and `lr=3e-4: Eff=69.2, Spec=93.7, FF=76.7`.
- The paper's LoRA appendix also notes that LoRA can require much stronger LR than full tuning.

Reason:

- If experiment 1 under-trains, this checks whether v2 needs stronger LoRA LR.
- If `efficacy_mean` becomes high while `ff_hard_pct` remains low, the likely issue is LoRA-vs-full scheme difference rather than training failure.

Command:

```bash
NUM_PROCS=2 GPU_IDS=0 OUTPUT_PREFIX=supp_phi3_arc_downproj_lr3e4 BATCH_SIZE=1 ADAPTER_GROUP_SIZE=1 EVAL_INTERVAL=1 LORA_RANK=16 LORA_ALPHA=32 LR=3e-4 bash v2/run_sharded_unlearn.sh --lora_target_modules down_proj
```

### 3. Paper-LoRA Rank Check

Parameters:

| Parameter | Value |
| --- | --- |
| Base setting | Same as experiment 2 |
| `LORA_RANK` | `32` |
| `LORA_ALPHA` | `32` |

Paper comparison:

- Paper LoRA appendix Table 9 sweeps rank in `{8, 32, 128}`, uses `lora_alpha=32`, and targets `down_proj`.
- The paper does not report Phi-3 + ARC-Challenge LoRA metrics, so this is a new supplement point.

Reason:

- Checks whether rank 16 is too capacity-limited.
- Rank 32 is the lowest paper-listed rank above the current v2 default and is a reasonable memory/performance compromise.

Command:

```bash
NUM_PROCS=2 GPU_IDS=0 OUTPUT_PREFIX=supp_phi3_arc_downproj_r32_lr3e4 BATCH_SIZE=1 ADAPTER_GROUP_SIZE=1 EVAL_INTERVAL=1 LORA_RANK=32 LORA_ALPHA=32 LR=3e-4 bash v2/run_sharded_unlearn.sh --lora_target_modules down_proj
```

### 4. Current-v2 Scope Control

Parameters:

| Parameter | Value |
| --- | --- |
| Base setting | Same as experiment 2 |
| `lora_target_modules` | empty, meaning all `nn.Linear` layers |
| Attention layers | Included if implemented as `nn.Linear` |

Paper comparison:

- This is not the paper LoRA setup.
- It is a v2 control to compare all-linear LoRA against the paper-style `down_proj` LoRA.

Reason:

- Historical v2 runs used all-linear LoRA.
- This checks whether modifying attention/all-linear layers changes `FF-HARD`, `FF-SOFT`, `Eff`, or `Spec`.

Command:

```bash
NUM_PROCS=2 GPU_IDS=0 OUTPUT_PREFIX=supp_phi3_arc_alllinear_lr3e4 BATCH_SIZE=1 ADAPTER_GROUP_SIZE=1 EVAL_INTERVAL=1 LORA_RANK=16 LORA_ALPHA=32 LR=3e-4 bash v2/run_sharded_unlearn.sh
```

## Result Collection

After each experiment finishes, run:

```bash
conda run -n pf-a100 python v2/summarize_results.py --results-root v2_outputs/final_results --outdir v2_outputs/reproduction/summary
```

Primary output:

```text
v2_outputs/reproduction/summary/v2_result_summary.csv
```

Core fields:

| Field | Meaning |
| --- | --- |
| `rows` | Number of completed adapter-step results |
| `unique_instances` | Number of source QA instances covered |
| `epoch0_rows` | Number of rows with epoch 0 baseline evaluation |
| `specificity_rows` | Number of rows with specificity evaluation |
| `new_cot_rows` | Number of rows with post-unlearning CoT |
| `ff_hard_pct` | Paper-style answer flip rate |
| `ff_soft_pct_mean` | Probability mass shifted away from initial answer |
| `efficacy_mean` | Reduction in target CoT-step probability |
| `specificity_mean` | Agreement with base model on held-out same-task examples |
| `loss_start_mean`, `loss_final_mean` | Training loss movement |

## Interpretation Rules

Use these thresholds as the first-pass diagnosis:

| Observation | Interpretation |
| --- | --- |
| `rows` is far below expected step count | Run incomplete, skipped, or crashed |
| `epoch0_rows != rows` | Cannot compute paper-style FF-HARD/FF-SOFT cleanly |
| `specificity_mean < 95` | Unlearning is too destructive for paper-style validity |
| `efficacy_mean` low and loss does not drop | Training strength or LoRA capacity is insufficient |
| `efficacy_mean` high, `specificity_mean` high, `ff_hard_pct` low | LoRA-FUR likely not equivalent to full/FF2 FUR |
| rank 32 improves strongly over rank 16 | v2 is capacity-limited |
| all-linear improves over down_proj but damages specificity | Attention/all-linear updates are stronger but less paper-aligned |

## Operational Notes

- If GPU OOM occurs, rerun the same command with `NUM_PROCS=1 GPU_IDS=0`; keep all experimental parameters unchanged.
- Do not reuse `OUTPUT_PREFIX` unless intentionally resuming a partially completed run.
- v2 deletes adapter weights after writing result rows by default; the `.out` files and summary CSV are the primary experiment artifacts.
- Current known paper targets for Phi-3 + ARC-Challenge are `Eff=40.8`, `Spec=99.5`, and `FUR/FF-HARD=39.1`; v2 LoRA results should be interpreted against those numbers, not treated as exact reproduction by default.
