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

### Loss extensions

The default v2 training path still uses `--method npo_KL`. Recent loss work adds optional controls for behavior-level and mechanism-level ablations:

```bash
--method npo_KL --forget_k_tokens 4
```

This limits the forget objective to the first `k` target CoT-step tokens. It is a stable baseline for blocking the start of a target reasoning step, but by itself it can still be string-level suppression.

```bash
--method npo_KL \
  --forget_k_tokens 4 \
  --repr_loss \
  --repr_lambda 0.1 \
  --repr_last_layers 4 \
  --repr_gamma 0.9 \
  --repr_auto_scale
```

This adds the current first-version mechanism loss: a layer-weighted hidden-state similarity penalty on the target step. `--repr_last_layers` selects the last layers, `--repr_gamma` downweights earlier selected layers, `--repr_k_tokens` can restrict the representation term to the first target tokens, and `--repr_auto_scale` rescales the auxiliary term against the current forget loss.

```bash
--method npo_KL \
  --causal_cot_loss \
  --causal_cot_lambda 0.1 \
  --causal_cot_margin 1.0 \
  --causal_cot_counterfactual remove_step \
  --causal_cot_answer correct \
  --causal_cot_auto_scale
```

This adds the Causal_CoT/FRODO-style reasoning-module loss adapted to the v2 causal-LM LoRA path. For each target step, v2 builds two answer prompts:

```text
full:           question + original CoT -> answer
counterfactual: question + CoT with target step removed -> same answer
```

The auxiliary objective is:

```text
L_IE = -log p(answer | question, original CoT)
L_MR = max(0, margin - (logp_full - logp_counterfactual))
L_causal_cot = lambda * (ie_lambda * L_IE + margin_lambda * L_MR)
```

`--causal_cot_auto_scale` rescales this auxiliary term against the current unlearning loss before applying `--causal_cot_lambda`. The implementation intentionally ports the FRODO reasoning-module objective rather than the seq2seq wrapper from `Causal_CoT/src/frodo.py`, because v2 trains causal-LM LoRA adapters on stepwise jobs.

Implementation entry points:

- `mechanistic_objectives.py`: masked loss, layer-weighted representation loss, and auxiliary scaling.
- `causal_cot_objectives.py`: FRODO-style answer-side IE and counterfactual margin loss.
- `v2/unlearn.py`: v2 stepwise integration and CLI flags.
- `../unlearn.py`: original pipeline integration and matching CLI flags.
- `../imporovement/05_mechanistic_unlearning_objective_plan.md`: design notes and next mechanisms.

Current status: implemented support is `first-k` forget, representation similarity penalty, and Causal_CoT/FRODO-style answer-margin training. Counterfactual representation loss and prototype contrastive loss are still not implemented.

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
