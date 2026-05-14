import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np


FILENAME_RE = re.compile(
    r"(?P<prefix>.+)_s=(?P<stepwise>True|False)_lr=(?P<lr>[^_]+)_rs=(?P<seed>\d+)_pos=(?P<pos>True|False)_ff2=(?P<ff2>True|False)(?:_(?P<suffix>.*))?\.out$"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize all v2 JSONL result files.")
    parser.add_argument("--results-root", default="v2_outputs/final_results")
    parser.add_argument("--outdir", default="v2_outputs/reproduction/summary")
    parser.add_argument("--pattern", default="*.out", help="Filename glob inside the result tree.")
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


def parse_filename(path: Path) -> dict:
    match = FILENAME_RE.match(path.name)
    if not match:
        return {
            "method": "",
            "run_type": "",
            "stepwise": "",
            "lr": "",
            "seed": "",
            "pos": "",
            "ff2": "",
            "suffix": "",
        }
    data = match.groupdict()
    prefix = data.pop("prefix")
    method, _, run_type = prefix.rpartition("_")
    data["method"] = method
    data["run_type"] = run_type
    data["suffix"] = data.get("suffix") or ""
    return data


def final_epoch_result(row: dict) -> tuple[str, dict]:
    results = row.get("unlearning_results") or {}
    if not results:
        return "", {}
    key = sorted(results.keys(), key=lambda value: int(value))[-1]
    return key, results[key]


def epoch0_result(row: dict) -> dict:
    return (row.get("unlearning_results") or {}).get("0") or {}


def epoch_predictions(row: dict) -> list[int]:
    results = row.get("unlearning_results") or {}
    predictions = []
    for key in sorted(results.keys(), key=lambda value: int(value)):
        prediction = prediction_from_probs(results[key])
        if prediction is not None:
            predictions.append(prediction)
    return predictions


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def median(values: list[float]) -> float:
    return float(np.median(values)) if values else float("nan")


def percentile(values: list[float], pct: float) -> float:
    return float(np.percentile(values, pct)) if values else float("nan")


def prediction_from_probs(result: dict) -> int | None:
    probs = result.get("probs")
    if not probs:
        return result.get("prediction")
    return int(np.argmax(probs))


def normalized_probs(result: dict) -> list[float]:
    probs = result.get("probs") or []
    total = float(sum(probs))
    if not probs or total <= 0.0:
        return []
    return [float(prob) / total for prob in probs]


def probability_drop_from_log_probs(baseline: dict, final: dict, key: str) -> tuple[float, float] | None:
    base_values = baseline.get(key)
    final_values = final.get(key)
    if not base_values or not final_values:
        return None
    base_log_prob = float(base_values[0])
    final_log_prob = float(final_values[0])
    base_prob = math.exp(base_log_prob)
    final_prob = math.exp(final_log_prob)
    if base_prob <= 0.0:
        return None
    return (1.0 - final_prob / base_prob) * 100.0, base_log_prob - final_log_prob


def answer_margin(probs: list[float]) -> float | None:
    if len(probs) < 2:
        return None
    sorted_probs = sorted(probs)
    return sorted_probs[-1] - sorted_probs[-2]


def summarize_file(path: Path, root: Path) -> dict:
    rows = load_jsonl(path)
    meta = parse_filename(path)
    rel = path.relative_to(root)
    dataset = rel.parts[0] if len(rel.parts) > 2 else ""
    model = rel.parts[1] if len(rel.parts) > 2 else ""

    answer_changed = []
    answer_changed_agree = []
    ff_soft = []
    ff_soft_agree = []
    answer_margin_drop = []
    answer_margin_drop_agree = []
    cot_prediction_changed = []
    efficacy = []
    cot_prob_drop = []
    cot_logprob_drop = []
    cot_step_prob_drop = []
    cot_step_logprob_drop = []
    specificity = []
    loss_start = []
    loss_final = []
    loss_min = []
    loss_drop = []
    final_epochs = []
    has_epoch0 = 0
    has_specificity = 0
    has_new_cot = 0
    question_final_changed = {}
    question_any_epoch_changed = {}
    agree_question_final_changed = {}
    agree_question_any_epoch_changed = {}
    agree_questions = set()

    for row in rows:
        final_key, final = final_epoch_result(row)
        if not final:
            continue
        baseline = epoch0_result(row)
        if baseline:
            has_epoch0 += 1
        final_pred = prediction_from_probs(final)
        initial_pred = prediction_from_probs(baseline) if baseline else row.get("prediction")
        if initial_pred is not None and final_pred is not None:
            answer_changed.append(initial_pred != final_pred)
            if row.get("prediction") == row.get("cot_prediction"):
                answer_changed_agree.append(initial_pred != final_pred)
            question = row.get("question")
            if question:
                question_final_changed[question] = (
                    question_final_changed.get(question, False) or (initial_pred != final_pred)
                )
                if row.get("prediction") == row.get("cot_prediction"):
                    agree_questions.add(question)
                    agree_question_final_changed[question] = (
                        agree_question_final_changed.get(question, False) or (initial_pred != final_pred)
                    )
        predictions = epoch_predictions(row)
        if len(predictions) >= 2:
            row_any_epoch_changed = any(prediction != predictions[0] for prediction in predictions)
            question = row.get("question")
            if question:
                question_any_epoch_changed[question] = (
                    question_any_epoch_changed.get(question, False) or row_any_epoch_changed
                )
                if row.get("prediction") == row.get("cot_prediction"):
                    agree_questions.add(question)
                    agree_question_any_epoch_changed[question] = (
                        agree_question_any_epoch_changed.get(question, False) or row_any_epoch_changed
                    )
        baseline_probs = normalized_probs(baseline)
        final_probs = normalized_probs(final)
        if baseline_probs and final_probs and len(baseline_probs) == len(final_probs):
            initial_answer = int(np.argmax(baseline_probs))
            # Paper FF-SOFT: probability mass shifted away from the initial answer.
            ff_soft.append((baseline_probs[initial_answer] - final_probs[initial_answer]) * 100.0)
            base_margin = answer_margin(baseline_probs)
            final_margin = answer_margin(final_probs)
            if base_margin is not None and final_margin is not None:
                answer_margin_drop.append((base_margin - final_margin) * 100.0)
            if row.get("prediction") == row.get("cot_prediction"):
                ff_soft_agree.append((baseline_probs[initial_answer] - final_probs[initial_answer]) * 100.0)
                if base_margin is not None and final_margin is not None:
                    answer_margin_drop_agree.append((base_margin - final_margin) * 100.0)
        cot_prediction = row.get("cot_prediction")
        if cot_prediction is not None and final_pred is not None:
            cot_prediction_changed.append(cot_prediction != final_pred)

        cot_drop = probability_drop_from_log_probs(baseline, final, "cot_prob")
        if cot_drop is not None:
            cot_prob_drop.append(cot_drop[0])
            cot_logprob_drop.append(cot_drop[1])

        step_drop = probability_drop_from_log_probs(baseline, final, "cot_step_prob")
        if step_drop is not None:
            efficacy.append(step_drop[0])
            cot_step_prob_drop.append(step_drop[0])
            cot_step_logprob_drop.append(step_drop[1])

        if baseline.get("specificity_preds") and final.get("specificity_preds"):
            base_preds = baseline["specificity_preds"]
            final_preds = final["specificity_preds"]
            if len(base_preds) == len(final_preds) and base_preds:
                has_specificity += 1
                specificity.append(
                    sum(a == b for a, b in zip(base_preds, final_preds)) / len(base_preds) * 100.0
                )

        if final.get("new_cot"):
            has_new_cot += 1

        history = row.get("adapter_training_history") or []
        losses = [entry.get("mean_loss") for entry in history if entry.get("mean_loss") is not None]
        if losses:
            loss_start.append(float(losses[0]))
            loss_final.append(float(losses[-1]))
            loss_min.append(float(min(losses)))
            loss_drop.append(float(losses[0] - losses[-1]))
        if final_key:
            final_epochs.append(int(final_key))

    return {
        "dataset": dataset,
        "model": model,
        **meta,
        "file": path.name,
        "rows": len(rows),
        "unique_instances": len({row.get("question") for row in rows}),
        "unique_adapter_ids": len({row.get("adapter_id") for row in rows if row.get("adapter_id")}),
        "agree_rows": len(answer_changed_agree),
        "epoch0_rows": has_epoch0,
        "specificity_rows": has_specificity,
        "new_cot_rows": has_new_cot,
        "final_epoch_max": max(final_epochs) if final_epochs else "",
        "ff_hard_pct": mean(answer_changed) * 100.0,
        "ff_hard_agree_pct": mean(answer_changed_agree) * 100.0,
        "question_final_ff_hard_pct": (
            sum(question_final_changed.values()) / len(question_final_changed) * 100.0
            if question_final_changed else float("nan")
        ),
        "question_any_epoch_ff_hard_pct": (
            sum(question_any_epoch_changed.values()) / len(question_any_epoch_changed) * 100.0
            if question_any_epoch_changed else float("nan")
        ),
        "agree_question_final_ff_hard_pct": (
            sum(agree_question_final_changed.get(question, False) for question in agree_questions)
            / len(agree_questions) * 100.0
            if agree_questions else float("nan")
        ),
        "agree_question_any_epoch_ff_hard_pct": (
            sum(agree_question_any_epoch_changed.get(question, False) for question in agree_questions)
            / len(agree_questions) * 100.0
            if agree_questions else float("nan")
        ),
        "ff_soft_pct_mean": mean(ff_soft),
        "ff_soft_pct_median": median(ff_soft),
        "ff_soft_agree_pct_mean": mean(ff_soft_agree),
        "answer_changed_pct": mean(answer_changed) * 100.0,
        "cot_prediction_changed_pct": mean(cot_prediction_changed) * 100.0,
        "answer_margin_drop_mean": mean(answer_margin_drop),
        "answer_margin_drop_agree_mean": mean(answer_margin_drop_agree),
        "efficacy_mean": mean(efficacy),
        "efficacy_median": median(efficacy),
        "efficacy_p10": percentile(efficacy, 10),
        "efficacy_p90": percentile(efficacy, 90),
        "cot_prob_drop_mean": mean(cot_prob_drop),
        "cot_logprob_drop_mean": mean(cot_logprob_drop),
        "cot_step_prob_drop_mean": mean(cot_step_prob_drop),
        "cot_step_prob_drop_median": median(cot_step_prob_drop),
        "cot_step_logprob_drop_mean": mean(cot_step_logprob_drop),
        "specificity_mean": mean(specificity),
        "loss_start_mean": mean(loss_start),
        "loss_final_mean": mean(loss_final),
        "loss_min_mean": mean(loss_min),
        "loss_drop_mean": mean(loss_drop),
        "loss_drop_median": median(loss_drop),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "model",
        "method",
        "run_type",
        "stepwise",
        "lr",
        "seed",
        "pos",
        "ff2",
        "suffix",
        "file",
        "rows",
        "unique_instances",
        "unique_adapter_ids",
        "agree_rows",
        "epoch0_rows",
        "specificity_rows",
        "new_cot_rows",
        "final_epoch_max",
        "ff_hard_pct",
        "ff_hard_agree_pct",
        "question_final_ff_hard_pct",
        "question_any_epoch_ff_hard_pct",
        "agree_question_final_ff_hard_pct",
        "agree_question_any_epoch_ff_hard_pct",
        "ff_soft_pct_mean",
        "ff_soft_pct_median",
        "ff_soft_agree_pct_mean",
        "answer_changed_pct",
        "cot_prediction_changed_pct",
        "answer_margin_drop_mean",
        "answer_margin_drop_agree_mean",
        "efficacy_mean",
        "efficacy_median",
        "efficacy_p10",
        "efficacy_p90",
        "cot_prob_drop_mean",
        "cot_logprob_drop_mean",
        "cot_step_prob_drop_mean",
        "cot_step_prob_drop_median",
        "cot_step_logprob_drop_mean",
        "specificity_mean",
        "loss_start_mean",
        "loss_final_mean",
        "loss_min_mean",
        "loss_drop_mean",
        "loss_drop_median",
    ]
    with path.open("w", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.results_root)
    outdir = Path(args.outdir)
    files = sorted(root.rglob(args.pattern))
    rows = [summarize_file(path, root) for path in files]
    rows.sort(key=lambda row: (row["dataset"], row["model"], float(row["lr"] or "nan"), row["file"]))

    outpath = outdir / "v2_result_summary.csv"
    write_csv(outpath, rows)
    summary = {
        "results_root": str(root),
        "files_found": len(files),
        "rows_written": len(rows),
        "csv": str(outpath),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
