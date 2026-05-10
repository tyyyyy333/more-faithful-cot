# Lightweight Reproduction

This repository is missing the large `results/` and `ablation/` artifacts needed for the full paper reproduction, so this lightweight path focuses on:

1. Reproducing the shipped annotation-study analysis.
2. Preparing offline aggregation scripts for the day the missing result files arrive.

## Environment

Create and activate a small conda environment:

```bash
conda create -n paramfaith python=3.10 pip -y
conda activate paramfaith
pip install -r requirements-analysis.txt
```

This intentionally avoids `torch`, `transformers`, and model downloads, because the target machine is a MacBook Air M2 with 16GB RAM.

## Reproduce Annotation Analysis

```bash
python reproduce_annotation_analysis.py
```

Outputs go to `reproduction/annotation/`.

## Reproduce Ablation and Full-Result Aggregation

Once the missing result directories are provided, place them under repo root as:

- `ablation/`
- `results/`

Then run:

```bash
python reproduce_ablation_stats.py --ablation-root ablation --results-root results
```

Outputs go to `reproduction/ablation/`.
