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


def summarize_file(path: Path, root: Path) -> dict:
    rows = load_jsonl(path)
    meta = parse_filename(path)
    rel = path.relative_to(root)
    dataset = rel.parts[0] if len(rel.parts) > 2 else ""
    model = rel.parts[1] if len(rel.parts) > 2 else ""

    answer_changed = []
    ff_soft = []
    cot_prediction_changed = []
    efficacy = []
    specificity = []
    loss_start = []
    loss_final = []
    loss_min = []
    loss_drop = []
    final_epochs = []
    has_epoch0 = 0
    has_specificity = 0
    has_new_cot = 0

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
        baseline_probs = normalized_probs(baseline)
        final_probs = normalized_probs(final)
        if baseline_probs and final_probs and len(baseline_probs) == len(final_probs):
            initial_answer = int(np.argmax(baseline_probs))
            # Paper FF-SOFT: probability mass shifted away from the initial answer.
            ff_soft.append((baseline_probs[initial_answer] - final_probs[initial_answer]) * 100.0)
        cot_prediction = row.get("cot_prediction")
        if cot_prediction is not None and final_pred is not None:
            cot_prediction_changed.append(cot_prediction != final_pred)

        if baseline.get("cot_step_prob") and final.get("cot_step_prob"):
            p0 = math.exp(baseline["cot_step_prob"][0])
            pf = math.exp(final["cot_step_prob"][0])
            efficacy.append((1.0 - pf / p0) * 100.0)

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
        "epoch0_rows": has_epoch0,
        "specificity_rows": has_specificity,
        "new_cot_rows": has_new_cot,
        "final_epoch_max": max(final_epochs) if final_epochs else "",
        "ff_hard_pct": mean(answer_changed) * 100.0,
        "ff_soft_pct_mean": mean(ff_soft),
        "ff_soft_pct_median": median(ff_soft),
        "answer_changed_pct": mean(answer_changed) * 100.0,
        "cot_prediction_changed_pct": mean(cot_prediction_changed) * 100.0,
        "efficacy_mean": mean(efficacy),
        "efficacy_median": median(efficacy),
        "efficacy_p10": percentile(efficacy, 10),
        "efficacy_p90": percentile(efficacy, 90),
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
        "epoch0_rows",
        "specificity_rows",
        "new_cot_rows",
        "final_epoch_max",
        "ff_hard_pct",
        "ff_soft_pct_mean",
        "ff_soft_pct_median",
        "answer_changed_pct",
        "cot_prediction_changed_pct",
        "efficacy_mean",
        "efficacy_median",
        "efficacy_p10",
        "efficacy_p90",
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
