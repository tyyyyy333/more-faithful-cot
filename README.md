# Codebase for the paper "Measuring Faithfulness of Chains of Thought by Unlearning Reasoning Steps"

Preprint: Tutek, M., Chaleshtori, F. H., Marasović, A., & Belinkov, Y. (2025). Measuring Faithfulness of Chains of Thought by Unlearning Reasoning Steps. [[arXiv]](https://arxiv.org/abs/2502.14829)

![Faithfulness by Unlearning Reasoning Steps](figures/fig1_v2.png "Faithfulness by Unlearning Reasoning Steps")

Codebase is given as-is, instructions pending.
Main file for running experiments is `unlearn.py`. The NPO method has been adapted from the [original repository](https://github.com/licong-lin/negative-preference-optimization).

Sample run script: `python unlearn.py --model_name meta-llama/Llama-3.2-3B-Instruct --strategy sentencize --stepwise --dataset sqa --lr 3e-05 --pos --ff2 --method npo_KL`

## v2 adapter reproduction

The v2 runner trains one LoRA adapter per unlearning target/step and writes each adapter plus a JSON record under `v2_outputs/`. For a fast single-GPU smoke test:

```bash
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

For independent multi-GPU adapter training, run one process per GPU. Each process loads its own base model and trains a disjoint shard of adapters; there is no DDP gradient synchronization and no parameter merge between GPUs. Use `--device_map none` so each process keeps its model on the CUDA device exposed by `CUDA_VISIBLE_DEVICES`.

The helper launcher below starts independent shard processes and writes one log file per shard. On a single 80GB A100, start with 2 processes and `ADAPTER_GROUP_SIZE=8`; 4 processes with rank-16 adapters can OOM because each process keeps its own Phi-3 copy and activation tensors. On multiple GPUs, set `GPU_IDS` to a comma-separated list and shards are assigned round-robin.

```bash
NUM_PROCS=2 GPU_IDS=0 BATCH_SIZE=8 ADAPTER_GROUP_SIZE=8 \
  bash v2/run_sharded_unlearn.sh
```

For a two-GPU run:

```bash
NUM_PROCS=4 GPU_IDS=0,1 BATCH_SIZE=8 ADAPTER_GROUP_SIZE=8 \
  bash v2/run_sharded_unlearn.sh
```

If this still OOMs, reduce `ADAPTER_GROUP_SIZE` to 4. If GPU utilization is low and there is free memory, increase `ADAPTER_GROUP_SIZE` gradually before increasing `NUM_PROCS`.

Extra arguments are passed through to `v2/unlearn.py`, so a small verification run can be launched with:

```bash
NUM_PROCS=2 GPU_IDS=0 OUTPUT_PREFIX=verify_2proc \
  bash v2/run_sharded_unlearn.sh --max_instances 2 --max_steps_per_instance 2
```

```bash
COMMON_ARGS="v2/unlearn.py \
  --model_name microsoft/Phi-3-mini-4k-instruct \
  --dataset arc-challenge \
  --strategy sentencize \
  --stepwise \
  --method npo_KL \
  --epochs 5 \
  --lr 1e-4 \
  --cot_limit 250 \
  --verify_size 20 \
  --retain_n 4 \
  --batch_size 1 \
  --eval_interval 1 \
  --device cuda \
  --device_map none \
  --adapter_group_size 2 \
  --lora_rank 16 \
  --lora_alpha 32"

CUDA_VISIBLE_DEVICES=0 conda run -n pf-a100 python $COMMON_ARGS --job_shard_count 2 --job_shard_index 0 &
CUDA_VISIBLE_DEVICES=1 conda run -n pf-a100 python $COMMON_ARGS --job_shard_count 2 --job_shard_index 1 &
wait
```

Sharded outputs are automatically separated by suffix. For two shards, result files are written to `v2_outputs/final_results/<dataset>/<model>/*_shard=0-of-2.out` and `*_shard=1-of-2.out`; adapter checkpoints and adapter records use matching suffixed directories under `v2_outputs/adapters/` and `v2_outputs/adapter_records/`.

In stepwise mode, the training unit is one CoT step, not one whole instance. A single instance with N segmented CoT steps creates N adapter jobs and therefore N independent adapters, with adapter ids ending in `_step_0`, `_step_1`, and so on. The sharding logic splits these step-level jobs across processes. Retain examples are pre-encoded once per job, then each dataset access randomly samples one valid retain example instead of always reusing the first valid retain.

## Paper graphs, result files and analysis notebooks

To recompute results, you need final & ablation result files (`results`,`ablations`) which are too large to share via git. Please send an email to me [\[here\]](mailto:martin.tutek@gmail.com) and I'll share the google drive links with you.

### Add mistake [Lanham et al, 2023](https://arxiv.org/abs/2307.13702)
We reuse the prompts from [Lanham et al](https://arxiv.org/abs/2307.13702) to add mistakes into CoT steps. A reproduction of this with GPT-4o-mini can be found in [Adding mistakes repro](Adding%20mistakes%20repro.ipynb). The minimal results of this setup can be found in [minimal_mistake_results](minimal_mistake_results).

### Annotation study

The annotation study data files, including all the per-model-dataset bins can be found in [annotation_data](annotation_data).
The code used to select instances for the study is in [Generate_annotation_data.ipynb](Generate_annotation_data.ipynb).

The full results of the annotation study can be fond in [annotation_results](annotation_results).
The follow up analysis can be found in [Annotation analysis.ipynb](Annotation%20analysis.ipynb).

### Post-unlearning CoT LLM-as-judge

The code using GPT-4o as a judge of whether CoTs have changed the answer they argue for before and after unlearning can be found in [CoT LLM as judge.ipynb](CoT%20LLM%20as%20judge.ipynb).
The LM judgements, along with the single-sentence explanations (which were not analysed in the paper) are in [LM_judge_cot](LM_judge_cot).

### Plots & tables
Most of the code used to generate plots and tables from the paper, along with the plots and tables themselves, can be found in [Ablations.ipynb](Ablations.ipynb) and [Generate_CoT_heatmaps.ipynb](Generate_CoT_heatmaps.ipynb).
