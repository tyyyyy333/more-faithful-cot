# v2 Training Experience, 2026-05-11

## Goal

Run the v2 stepwise unlearning pipeline at a stronger setting, validate whether multi-process sharding improves throughput, and check whether the resulting adapters actually change faithfulness-related behavior.

## Final Configuration

```bash
NUM_PROCS=2 GPU_IDS=0 \
  OUTPUT_PREFIX=strong_2proc \
  EPOCHS=5 LR=1e-4 \
  BATCH_SIZE=8 ADAPTER_GROUP_SIZE=8 \
  LORA_RANK=16 LORA_ALPHA=32 \
  bash v2/run_sharded_unlearn.sh
```

Dataset/model:
- `microsoft/Phi-3-mini-4k-instruct`
- `arc-challenge`
- `sentencize`
- `stepwise`
- `npo_KL`

## What Happened

- The run produced partially completed results before one shard hit CUDA OOM.
- `strong_2proc_s1` completed almost the whole shard.
- `strong_2proc_s0` stopped early at `group 14/114` with:
  - `torch.cuda.OutOfMemoryError: Tried to allocate 444.00 MiB`
- The saved rows are structurally valid JSONL; there was no corruption in the written results.

## Saved Output

- Final results:
  - `v2_outputs/final_results/arc-challenge/Phi-3/npo_KL_sentencize_s=True_lr=0.0001_rs=1001_pos=False_ff2=False_strong_2proc_s0.out`
  - `v2_outputs/final_results/arc-challenge/Phi-3/npo_KL_sentencize_s=True_lr=0.0001_rs=1001_pos=False_ff2=False_strong_2proc_s1.out`
- Adapter records:
  - `v2_outputs/adapter_records/arc-challenge/Phi-3/npo_KL_sentencize_s=True_lr=0.0001_rs=1001_pos=False_ff2=False_strong_2proc_s0/`
  - `v2_outputs/adapter_records/arc-challenge/Phi-3/npo_KL_sentencize_s=True_lr=0.0001_rs=1001_pos=False_ff2=False_strong_2proc_s1/`

## Observed Effect

Across the written portion of the run:
- `1010` rows total
- `cot_prediction` changed in `102/1010 = 10.10%`
- final answer prediction changed in `17/1010 = 1.68%`
- `905/1010` rows kept both base and cot prediction unchanged

Shard breakdown:
- `s0`: `104` rows, `7` cot prediction changes, `1` final prediction change
- `s1`: `906` rows, `95` cot prediction changes, `16` final prediction changes

Interpretation:
- The run produced real signal.
- The effect is much stronger on CoT-side outputs than on final answer selection.
- This configuration is still not strong enough to move the final answer often.

## Operational Notes

- The repo now deletes adapter `.pt` files after their results are stored unless `--keep_adapter_weights` is set.
- The earlier disk-quota failure was caused by retaining all adapter checkpoints; the current default avoids that, but the present run still OOMed on GPU memory.
- For the next attempt, lower `NUM_PROCS` or `ADAPTER_GROUP_SIZE`, or both.

