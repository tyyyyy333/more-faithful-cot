# v2 runner

This directory contains the stepwise multi-adapter rewrite of the original unlearning pipeline.

## What v2 does

- One CoT step becomes one training job.
- One job trains one LoRA adapter.
- Jobs can be sharded across independent Python processes.
- There is no DDP, no gradient synchronization, and no parameter merge across GPUs.
- Adapter checkpoints are deleted after their result row is written unless `--keep_adapter_weights` is set.

In stepwise mode, the unit of work is:

```text
instance -> segmented CoT -> step_idx -> adapter_id
```

The step id is recorded in the JSONL output as `step_idx` and `adapter_id`. The result filename is only a run-level suffix; it is not a step-level identifier.

## Training

Fast smoke test:

```bash
cd /cpfs01/projects-HDD/cfff-d9e1999a777f_HDD/tyg_23300200022/PJ/more-faithful-cot && \
conda run -n pf-a100 python v2/unlearn.py \
  --model_name microsoft/Phi-3-mini-4k-instruct \
  --dataset arc-challenge \
  --strategy sentencize \
  --stepwise \
  --method npo_KL \
  --epochs 1 \
  --lr 2e-5 \
  --cot_limit 16 \
  --verify_size 2 \
  --retain_n 4 \
  --batch_size 1 \
  --max_instances 1 \
  --max_steps_per_instance 1 \
  --eval_interval 1 \
  --skip_specificity \
  --skip_new_cot \
  --skip_initial_eval \
  --device cuda \
  --device_map none \
  --adapter_group_size 1 \
  --lora_rank 8
```

Multi-process launcher:

```bash
cd /cpfs01/projects-HDD/cfff-d9e1999a777f_HDD/tyg_23300200022/PJ/more-faithful-cot && \
NUM_PROCS=2 GPU_IDS=0 OUTPUT_PREFIX=compare_original_like_mp \
EPOCHS=5 LR=1e-4 BATCH_SIZE=1 ADAPTER_GROUP_SIZE=8 LORA_RANK=8 LORA_ALPHA=16 \
bash v2/run_sharded_unlearn.sh \
  --max_instances 2 \
  --max_steps_per_instance 2
```

If you have two visible GPUs:

```bash
NUM_PROCS=2 GPU_IDS=0,1 OUTPUT_PREFIX=compare_original_like_mp \
EPOCHS=5 LR=1e-4 BATCH_SIZE=1 ADAPTER_GROUP_SIZE=8 LORA_RANK=8 LORA_ALPHA=16 \
bash v2/run_sharded_unlearn.sh \
  --max_instances 2 \
  --max_steps_per_instance 2
```

Notes:

- `--device_map none` keeps each process on the CUDA device exposed by `CUDA_VISIBLE_DEVICES`.
- `--ff2` is not supported in the current multi-adapter stepwise flow.
- `--pos` is supported.
- `retain_n` is random-sampled from valid retain examples after pre-encoding.

## Result layout

Outputs are written under `v2_outputs/`:

- `v2_outputs/final_results/<dataset>/<model>/*.out`
- `v2_outputs/adapter_records/<dataset>/<model>/*.json`
- `v2_outputs/adapters/<dataset>/<model>/*.pt`

Sharded runs append a suffix such as `shard=0-of-2` to keep files separate.

The JSONL result rows contain:

- `adapter_id`
- `step_idx`
- `global_job_index`
- `adapter_training_history`
- `unlearning_results`

## Metrics

The main metrics are computed in `v2/stats.py`:

- `faithfulness`: fraction of unique questions whose answer prediction changes at least once during unlearning.
- `efficacy`: reduction in probability of the target CoT step, computed from `cot_step_prob` in stepwise mode.
- `specificity`: agreement on held-out same-task examples, computed from `specificity_preds`.
- `n_instances`: number of unique questions.
- `n_cot_steps`: number of step-level jobs.

Interpretation:

- Higher `faithfulness` means the target behavior changes more often.
- Higher `efficacy` means the target step is suppressed more strongly.
- Higher `specificity` means less collateral damage on held-out examples.

## Summaries

To aggregate v2 results into a CSV table:

```bash
cd /cpfs01/projects-HDD/cfff-d9e1999a777f_HDD/tyg_23300200022/PJ/more-faithful-cot && \
conda run -n pf-a100 python v2/reproduce_ablation_stats.py \
  --results-root v2_outputs/final_results \
  --ablation-root v2_outputs/ablation \
  --outdir v2_outputs/reproduction/ablation \
  --method npo_KL \
  --run-type sentencize \
  --seed 1001 \
  --no-pos \
  --no-ff2
```

This writes:

- `v2_outputs/reproduction/ablation/summary.json`
- `v2_outputs/reproduction/ablation/best_full_results.csv`
- `v2_outputs/reproduction/ablation/ablation_stats.csv` if ablation files exist

To compare against the original pipeline, run the same aggregator on `final_results/` instead of `v2_outputs/final_results/`.

## Practical warnings

- The current multi-adapter stepwise flow is the intended v2 path.
- `strong_2proc` and similar suffixes are run labels, not step labels.
- If the run OOMs, reduce `BATCH_SIZE`, then `ADAPTER_GROUP_SIZE`, then `NUM_PROCS`.
