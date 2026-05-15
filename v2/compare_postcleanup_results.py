#!/usr/bin/env python
"""Compare the post-cleanup v2 LoRA experiments.

This script reads active JSONL final-result files and reports metrics using a
true union over questions across shards. It is intended to reproduce the
post-cleanup comparison table in docs/v2_experiment_registry.md.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean


DEFAULT_GROUPS = {
    "downproj_r32_lr5e4": "*downproj_r32_lr5e4_posfix_gs2_s*.out",
    "alllinear_lr3e4": "*alllinear_lr3e4_postcleanup_s*.out",
    "alllinear_r32_lr5e4": "*alllinear_r32_lr5e4_postcleanup_s*.out",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare post-cleanup v2 LoRA experiment groups."
    )
    parser.add_argument(
        "--results-dir",
        default="v2_outputs/final_results/arc-challenge/Phi-3",
        help="Directory containing active .out JSONL result files.",
    )
    return parser


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as infile:
        for line_no, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
    return rows


def prediction(result: dict) -> int | None:
    probs = result.get("probs") or []
    if probs:
        return int(max(range(len(probs)), key=lambda idx: probs[idx]))
    pred = result.get("prediction")
    return int(pred) if pred is not None else None


def sorted_epoch_results(row: dict) -> list[dict]:
    results = row.get("unlearning_results") or {}
    return [results[key] for key in sorted(results, key=lambda value: int(value))]


def step_efficacy(baseline: dict, final: dict) -> float | None:
    base_values = baseline.get("cot_step_prob")
    final_values = final.get("cot_step_prob")
    if not base_values or not final_values:
        return None
    base_prob = math.exp(float(base_values[0]))
    final_prob = math.exp(float(final_values[0]))
    if base_prob <= 0.0:
        return None
    return (1.0 - final_prob / base_prob) * 100.0


def specificity(baseline: dict, final: dict) -> float | None:
    base_preds = baseline.get("specificity_preds") or []
    final_preds = final.get("specificity_preds") or []
    if not base_preds or len(base_preds) != len(final_preds):
        return None
    unchanged = sum(left == right for left, right in zip(base_preds, final_preds))
    return unchanged / len(base_preds) * 100.0


def final_loss(row: dict) -> float | None:
    history = row.get("adapter_training_history") or []
    losses = [entry.get("mean_loss") for entry in history if entry.get("mean_loss") is not None]
    return float(losses[-1]) if losses else None


def summarize_group(results_dir: Path, pattern: str) -> dict:
    files = sorted(results_dir.glob(pattern))
    rows = []
    for path in files:
        rows.extend(load_jsonl(path))

    valid_rows = 0
    agree_rows = 0
    row_final_flips = 0
    row_any_epoch_flips = 0
    question_final_flips: dict[str, bool] = {}
    question_any_epoch_flips: dict[str, bool] = {}
    agree_questions = set()
    agree_question_final_flips: dict[str, bool] = {}
    efficacies = []
    specificities = []
    final_losses = []

    for row in rows:
        epochs = sorted_epoch_results(row)
        if len(epochs) < 2:
            continue
        preds = [prediction(epoch) for epoch in epochs]
        if any(pred is None for pred in preds):
            continue

        baseline = epochs[0]
        final = epochs[-1]
        question = row.get("question")
        if not question:
            continue

        initial_pred = preds[0]
        final_flip = preds[-1] != initial_pred
        any_epoch_flip = any(pred != initial_pred for pred in preds[1:])
        valid_rows += 1
        row_final_flips += int(final_flip)
        row_any_epoch_flips += int(any_epoch_flip)
        question_final_flips[question] = question_final_flips.get(question, False) or final_flip
        question_any_epoch_flips[question] = (
            question_any_epoch_flips.get(question, False) or any_epoch_flip
        )

        if row.get("prediction") == row.get("cot_prediction"):
            agree_rows += 1
            agree_questions.add(question)
            agree_question_final_flips[question] = (
                agree_question_final_flips.get(question, False) or final_flip
            )

        efficacy = step_efficacy(baseline, final)
        if efficacy is not None:
            efficacies.append(efficacy)

        spec = specificity(baseline, final)
        if spec is not None:
            specificities.append(spec)

        loss = final_loss(row)
        if loss is not None:
            final_losses.append(loss)

    question_count = len(question_final_flips)
    agree_question_count = len(agree_questions)
    return {
        "files": len(files),
        "rows": valid_rows,
        "questions": question_count,
        "agree_rows": agree_rows,
        "agree_questions": agree_question_count,
        "row_final_ff_hard_pct": row_final_flips / valid_rows * 100.0,
        "row_any_epoch_ff_hard_pct": row_any_epoch_flips / valid_rows * 100.0,
        "question_final_ff_hard_pct": (
            sum(question_final_flips.values()) / question_count * 100.0
        ),
        "question_any_epoch_ff_hard_pct": (
            sum(question_any_epoch_flips.values()) / question_count * 100.0
        ),
        "agree_question_final_ff_hard_pct": (
            sum(agree_question_final_flips.get(question, False) for question in agree_questions)
            / agree_question_count
            * 100.0
        ),
        "efficacy_mean": mean(efficacies),
        "specificity_mean": mean(specificities),
        "loss_final_mean": mean(final_losses),
    }


def main() -> None:
    args = build_parser().parse_args()
    results_dir = Path(args.results_dir)

    headers = [
        "group",
        "files",
        "rows",
        "questions",
        "agree_rows",
        "row_final_ff_hard_pct",
        "question_final_ff_hard_pct",
        "question_any_epoch_ff_hard_pct",
        "agree_question_final_ff_hard_pct",
        "efficacy_mean",
        "specificity_mean",
        "loss_final_mean",
    ]
    print("\t".join(headers))
    for group, pattern in DEFAULT_GROUPS.items():
        summary = summarize_group(results_dir, pattern)
        values = [group]
        for header in headers[1:]:
            value = summary[header]
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        print("\t".join(values))


if __name__ == "__main__":
    main()
