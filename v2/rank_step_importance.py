#!/usr/bin/env python
"""Rank CoT steps by counterfactual answer effect.

For each (question, step), compare the base model's answer distribution under:

  full CoT prompt
  removed-step CoT prompt

The resulting CSV/JSONL is intended to select steps whose removal actually
changes answer evidence before running unlearning experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from data import load_or_generate_dataset_cots, model_name_dict
from dataload import DATASETS
from evaluate import ANSWER_LETTERS
from models import load_model_and_tokenizer, model_input_device
from util import set_random_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rank CoT steps by removed-step counterfactual answer effect."
    )
    parser.add_argument("--model_name", default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--dataset", default="arc-challenge")
    parser.add_argument("--strategy", default="sentencize", choices=["sentencize"])
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cot_limit", type=int, default=250)
    parser.add_argument("--max_instances", type=int, default=0)
    parser.add_argument("--max_steps_per_instance", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="none")
    parser.add_argument(
        "--outdir",
        default="v2_outputs/step_importance",
        help="Directory for CSV/JSONL ranked outputs.",
    )
    parser.add_argument(
        "--output_suffix",
        default="",
        help="Optional suffix added to output filenames.",
    )
    parser.add_argument(
        "--min_steps",
        type=int,
        default=1,
        help="Skip instances with fewer than this many segmented CoT steps.",
    )
    return parser


def answer_prompt(dh, target: dict[str, Any], cot_text: str) -> str:
    cot_prefix = dh.make_cot_prompt(target["raw_instance"])
    return dh.make_answer_prompt(cot_prefix + cot_text)


def top_margin(probs: list[float]) -> float:
    if len(probs) < 2:
        return 0.0
    values = sorted((float(value) for value in probs), reverse=True)
    return values[0] - values[1]


def argmax(values: list[float]) -> int:
    return int(max(range(len(values)), key=lambda idx: values[idx]))


def answer_index_for_letter(letters: list[str], letter: str | None) -> int | None:
    if letter is None:
        return None
    try:
        return letters.index(letter)
    except ValueError:
        return None


def build_jobs(cot_data: list[dict[str, Any]], dh, args) -> list[dict[str, Any]]:
    targets = cot_data[: args.max_instances] if args.max_instances > 0 else cot_data
    jobs = []
    for target_index, target in enumerate(targets):
        segmented = target.get("segmented_cot") or []
        if len(segmented) < args.min_steps:
            continue
        step_count = len(segmented)
        if args.max_steps_per_instance > 0:
            step_count = min(step_count, args.max_steps_per_instance)
        for step_idx in range(step_count):
            removed_steps = [
                step for idx, step in enumerate(segmented) if idx != step_idx
            ]
            removed_cot = "\n".join(removed_steps)
            jobs.append({
                "target_index": target_index,
                "target": target,
                "step_idx": step_idx,
                "step_text": segmented[step_idx],
                "full_prompt": answer_prompt(dh, target, target["cot"]),
                "removed_prompt": answer_prompt(dh, target, removed_cot),
            })
    return jobs


def score_prompts(model, tokenizer, prompts: list[str], answer_indices: list[int]) -> list[list[float]]:
    device = model_input_device(model)
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    encoded = tokenizer(
        prompts,
        padding=True,
        add_special_tokens=False,
        return_tensors="pt",
    )
    tokenizer.padding_side = old_padding_side

    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
    last_positions = attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(input_ids.shape[0], device=device)
    logits = outputs.logits[batch_indices, last_positions, :]
    probs = torch.softmax(logits[:, answer_indices], dim=-1)
    return probs.detach().cpu().float().numpy().tolist()


def score_jobs(model, tokenizer, jobs: list[dict[str, Any]], dh, batch_size: int) -> list[dict[str, Any]]:
    rows = []
    grouped: dict[tuple[int, ...], list[dict[str, Any]]] = {}
    for job in jobs:
        letters = list(dh.get_answer_letters(job["target"]["raw_instance"]))
        answer_indices = tuple(
            tokenizer.encode(letter, add_special_tokens=False)[0]
            for letter in letters
        )
        grouped.setdefault(answer_indices, []).append(job)

    for answer_indices, group_jobs in grouped.items():
        for start in tqdm(range(0, len(group_jobs), batch_size), desc="Scoring steps"):
            batch_jobs = group_jobs[start:start + batch_size]
            prompts = []
            for job in batch_jobs:
                prompts.append(job["full_prompt"])
                prompts.append(job["removed_prompt"])

            prompt_probs = score_prompts(model, tokenizer, prompts, list(answer_indices))
            for local_idx, job in enumerate(batch_jobs):
                full_probs = prompt_probs[2 * local_idx]
                removed_probs = prompt_probs[2 * local_idx + 1]
                full_pred = argmax(full_probs)
                removed_pred = argmax(removed_probs)
                letters = list(dh.get_answer_letters(job["target"]["raw_instance"]))
                correct_idx = answer_index_for_letter(
                    letters,
                    job["target"].get("correct_letter"),
                )
                cot_pred = int(np.argmax(job["target"].get("cot_probs", full_probs)))
                nocot_pred = int(np.argmax(job["target"].get("nocot_probs", full_probs)))
                reference_idx = cot_pred if 0 <= cot_pred < len(full_probs) else full_pred

                full_reference_prob = float(full_probs[reference_idx])
                removed_reference_prob = float(removed_probs[reference_idx])
                reference_prob_drop = full_reference_prob - removed_reference_prob
                margin_drop = top_margin(full_probs) - top_margin(removed_probs)
                correct_prob_drop = ""
                if correct_idx is not None and correct_idx < len(full_probs):
                    correct_prob_drop = (
                        float(full_probs[correct_idx]) - float(removed_probs[correct_idx])
                    )

                rows.append({
                    "id": job["target"].get("id", ""),
                    "question": job["target"].get("question", ""),
                    "target_index": job["target_index"],
                    "step_idx": job["step_idx"],
                    "step_text": job["step_text"],
                    "step_token_count": len(tokenizer.encode(job["step_text"], add_special_tokens=False)),
                    "answer_letters": "".join(letters),
                    "correct_letter": job["target"].get("correct_letter", ""),
                    "cot_prediction": cot_pred,
                    "nocot_prediction": nocot_pred,
                    "full_prediction": full_pred,
                    "removed_prediction": removed_pred,
                    "removed_flips_full_prediction": int(removed_pred != full_pred),
                    "removed_flips_cot_prediction": int(removed_pred != cot_pred),
                    "reference_prediction": reference_idx,
                    "full_reference_prob": full_reference_prob,
                    "removed_reference_prob": removed_reference_prob,
                    "reference_prob_drop": reference_prob_drop,
                    "full_margin": top_margin(full_probs),
                    "removed_margin": top_margin(removed_probs),
                    "margin_drop": margin_drop,
                    "correct_prob_drop": correct_prob_drop,
                    "full_probs": full_probs,
                    "removed_probs": removed_probs,
                    "importance_score": reference_prob_drop,
                })
    rows.sort(
        key=lambda row: (
            float(row["importance_score"]),
            float(row["margin_drop"]),
            int(row["removed_flips_cot_prediction"]),
        ),
        reverse=True,
    )
    return rows


def write_outputs(rows: list[dict[str, Any]], args) -> tuple[Path, Path, Path]:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    short_model = model_name_dict.get(args.model_name.split("/")[-1], args.model_name.split("/")[-1])
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    stem = f"{args.dataset}_{short_model}_step_importance{suffix}"
    csv_path = outdir / f"{stem}.csv"
    jsonl_path = outdir / f"{stem}.jsonl"
    summary_path = outdir / f"{stem}_summary.json"

    csv_fields = [
        "id",
        "target_index",
        "step_idx",
        "step_token_count",
        "correct_letter",
        "cot_prediction",
        "nocot_prediction",
        "full_prediction",
        "removed_prediction",
        "removed_flips_full_prediction",
        "removed_flips_cot_prediction",
        "reference_prediction",
        "full_reference_prob",
        "removed_reference_prob",
        "reference_prob_drop",
        "full_margin",
        "removed_margin",
        "margin_drop",
        "correct_prob_drop",
        "importance_score",
        "question",
        "step_text",
    ]
    with csv_path.open("w", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            record = dict(row)
            for key in ("full_probs", "removed_probs"):
                record.pop(key, None)
            writer.writerow({key: record.get(key, "") for key in csv_fields})

    with jsonl_path.open("w") as outfile:
        for row in rows:
            outfile.write(json.dumps(row, ensure_ascii=False) + "\n")

    scores = [float(row["importance_score"]) for row in rows]
    summary = {
        "rows": len(rows),
        "questions": len({row["id"] for row in rows}),
        "removed_flips_full_prediction": sum(row["removed_flips_full_prediction"] for row in rows),
        "removed_flips_cot_prediction": sum(row["removed_flips_cot_prediction"] for row in rows),
        "importance_score_mean": float(np.mean(scores)) if scores else None,
        "importance_score_median": float(np.median(scores)) if scores else None,
        "importance_score_p90": float(np.percentile(scores, 90)) if scores else None,
        "importance_score_max": float(np.max(scores)) if scores else None,
        "top10": [
            {
                "id": row["id"],
                "step_idx": row["step_idx"],
                "importance_score": row["importance_score"],
                "margin_drop": row["margin_drop"],
                "removed_flips_cot_prediction": row["removed_flips_cot_prediction"],
                "step_text": row["step_text"],
            }
            for row in rows[:10]
        ],
    }
    with summary_path.open("w") as outfile:
        json.dump(summary, outfile, ensure_ascii=False, indent=2)

    return csv_path, jsonl_path, summary_path


def main() -> None:
    args = build_parser().parse_args()
    set_random_seed(args.seed)
    random.seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        device_pref=args.device,
        device_map=args.device_map,
    )
    model.eval()
    dh = DATASETS[args.dataset]
    cot_data = load_or_generate_dataset_cots(
        model_id=args.model_name,
        tokenizer=tokenizer,
        dataset_id=args.dataset,
        force_generate=False,
        sentencize=args.strategy == "sentencize",
        temperature=args.temperature,
        seed=args.seed,
        max_instances=args.cot_limit,
        device_pref=args.device,
        model=model,
    )
    jobs = build_jobs(cot_data, dh, args)
    print(f"Scoring {len(jobs)} step jobs from {len(cot_data)} CoT instances")
    rows = score_jobs(model, tokenizer, jobs, dh, batch_size=args.batch_size)
    csv_path, jsonl_path, summary_path = write_outputs(rows, args)
    print(f"Wrote {csv_path}")
    print(f"Wrote {jsonl_path}")
    print(f"Wrote {summary_path}")
    if rows:
        print("Top steps:")
        for row in rows[:10]:
            print(
                f"{row['id']} step={row['step_idx']} "
                f"score={row['importance_score']:.6f} "
                f"margin_drop={row['margin_drop']:.6f} "
                f"flip={row['removed_flips_cot_prediction']} "
                f"text={row['step_text'][:120]!r}"
            )


if __name__ == "__main__":
    main()
