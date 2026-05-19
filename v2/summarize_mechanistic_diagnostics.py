#!/usr/bin/env python
"""Summarize v2 mechanistic diagnostics embedded in result JSONL files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_RESULTS_DIR = "v2_outputs/final_results/arc-challenge/Phi-3"
DEFAULT_OUTDIR = "v2_outputs/reproduction/mechanistic"
SERIES_KEYS = {
    "repr_step_answer": ("representation", "step_answer_cosine_by_layer"),
    "repr_answer_removed": ("representation", "answer_removed_cosine_by_layer"),
    "attn_answer_to_step": ("attention", "answer_to_step_mass_by_layer"),
    "attn_answer_to_prefix": ("attention", "answer_to_prefix_mass_by_layer"),
    "attn_answer_to_post_step": ("attention", "answer_to_post_step_mass_by_layer"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate before/after hidden-state and attention diagnostics."
    )
    parser.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing v2 .out JSONL result files.",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        default=[],
        help="Glob pattern for files to include. May be repeated. Defaults to *.out.",
    )
    parser.add_argument(
        "--outdir",
        default=DEFAULT_OUTDIR,
        help="Directory for mechanistic_rows.csv and mechanistic_summary.tsv.",
    )
    return parser


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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


def infer_group(path: Path) -> str:
    stem = path.stem
    if "_ff2=False_" in stem:
        stem = stem.split("_ff2=False_", 1)[1]
    stem = re.sub(r"_s\d+$", "", stem)
    return stem


def sorted_epoch_items(row: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    results = row.get("unlearning_results") or {}
    items = []
    for key, value in results.items():
        try:
            items.append((int(key), value))
        except (TypeError, ValueError):
            continue
    return sorted(items, key=lambda item: item[0])


def prediction(result: dict[str, Any]) -> int | None:
    probs = result.get("probs") or []
    if probs:
        return int(max(range(len(probs)), key=lambda idx: probs[idx]))
    pred = result.get("prediction")
    return int(pred) if pred is not None else None


def step_efficacy(baseline: dict[str, Any], final: dict[str, Any]) -> float | None:
    base_values = baseline.get("cot_step_prob")
    final_values = final.get("cot_step_prob")
    if not base_values or not final_values:
        return None
    base_prob = math.exp(float(base_values[0]))
    final_prob = math.exp(float(final_values[0]))
    if base_prob <= 0.0:
        return None
    return (1.0 - final_prob / base_prob) * 100.0


def specificity(baseline: dict[str, Any], final: dict[str, Any]) -> float | None:
    base_preds = baseline.get("specificity_preds") or []
    final_preds = final.get("specificity_preds") or []
    if not base_preds or len(base_preds) != len(final_preds):
        return None
    unchanged = sum(left == right for left, right in zip(base_preds, final_preds))
    return unchanged / len(base_preds) * 100.0


def series_mean(values: Any) -> float | None:
    if not isinstance(values, list) or not values:
        return None
    numeric = [float(value) for value in values]
    return mean(numeric)


def series_last_k_mean(values: Any, k: int = 4) -> float | None:
    if not isinstance(values, list) or not values:
        return None
    numeric = [float(value) for value in values[-k:]]
    return mean(numeric)


def diagnostic_series(diag: dict[str, Any] | None, short_key: str) -> Any:
    if not diag:
        return None
    section, key = SERIES_KEYS[short_key]
    return diag.get(section, {}).get(key)


def row_summary(path: Path, row: dict[str, Any]) -> dict[str, Any] | None:
    epochs = sorted_epoch_items(row)
    if len(epochs) < 2:
        return None
    baseline = epochs[0][1]
    final = epochs[-1][1]
    before_diag = baseline.get("mechanistic_diagnostics")
    after_diag = final.get("mechanistic_diagnostics")
    if not before_diag or not after_diag:
        return None

    baseline_pred = prediction(baseline)
    final_pred = prediction(final)
    if baseline_pred is None or final_pred is None:
        return None

    result = {
        "group": infer_group(path),
        "file": path.name,
        "id": row.get("id", ""),
        "step_idx": row.get("step_idx", ""),
        "question": row.get("question", ""),
        "baseline_epoch": epochs[0][0],
        "final_epoch": epochs[-1][0],
        "final_flip": int(final_pred != baseline_pred),
        "efficacy": step_efficacy(baseline, final),
        "specificity": specificity(baseline, final),
        "warnings_before": "; ".join(before_diag.get("warnings", [])),
        "warnings_after": "; ".join(after_diag.get("warnings", [])),
    }

    for short_key in SERIES_KEYS:
        before_values = diagnostic_series(before_diag, short_key)
        after_values = diagnostic_series(after_diag, short_key)
        before_mean = series_mean(before_values)
        after_mean = series_mean(after_values)
        before_last4 = series_last_k_mean(before_values)
        after_last4 = series_last_k_mean(after_values)
        result[f"{short_key}_before_mean"] = before_mean
        result[f"{short_key}_after_mean"] = after_mean
        result[f"{short_key}_delta_mean"] = (
            after_mean - before_mean
            if before_mean is not None and after_mean is not None
            else None
        )
        result[f"{short_key}_before_last4"] = before_last4
        result[f"{short_key}_after_last4"] = after_last4
        result[f"{short_key}_delta_last4"] = (
            after_last4 - before_last4
            if before_last4 is not None and after_last4 is not None
            else None
        )
    return result


def safe_mean(values: list[Any]) -> float | None:
    numeric = [float(value) for value in values if value is not None and value != ""]
    return mean(numeric) if numeric else None


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {
        "files": len({row["file"] for row in rows}),
        "rows": len(rows),
        "questions": len({row["question"] for row in rows if row.get("question")}),
        "final_flip_pct": safe_mean([row["final_flip"] for row in rows]),
        "efficacy_mean": safe_mean([row["efficacy"] for row in rows]),
        "specificity_mean": safe_mean([row["specificity"] for row in rows]),
    }
    if out["final_flip_pct"] is not None:
        out["final_flip_pct"] *= 100.0
    for short_key in SERIES_KEYS:
        out[f"{short_key}_delta_mean"] = safe_mean(
            [row[f"{short_key}_delta_mean"] for row in rows]
        )
        out[f"{short_key}_delta_last4"] = safe_mean(
            [row[f"{short_key}_delta_last4"] for row in rows]
        )
        out[f"{short_key}_after_last4"] = safe_mean(
            [row[f"{short_key}_after_last4"] for row in rows]
        )
    return out


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main() -> None:
    args = build_parser().parse_args()
    results_dir = Path(args.results_dir)
    patterns = args.pattern or ["*.out"]
    files = []
    for pattern in patterns:
        files.extend(sorted(results_dir.glob(pattern)))
    files = sorted(set(files))

    rows = []
    for path in files:
        for row in load_jsonl(path):
            summary = row_summary(path, row)
            if summary is not None:
                rows.append(summary)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    row_headers = [
        "group",
        "file",
        "id",
        "step_idx",
        "question",
        "baseline_epoch",
        "final_epoch",
        "final_flip",
        "efficacy",
        "specificity",
    ]
    for short_key in SERIES_KEYS:
        row_headers.extend([
            f"{short_key}_before_mean",
            f"{short_key}_after_mean",
            f"{short_key}_delta_mean",
            f"{short_key}_before_last4",
            f"{short_key}_after_last4",
            f"{short_key}_delta_last4",
        ])
    row_headers.extend(["warnings_before", "warnings_after"])

    rows_path = outdir / "mechanistic_rows.csv"
    with rows_path.open("w", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=row_headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_value(row.get(key)) for key in row_headers})

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["group"], []).append(row)

    summary_headers = [
        "group",
        "files",
        "rows",
        "questions",
        "final_flip_pct",
        "efficacy_mean",
        "specificity_mean",
    ]
    for short_key in SERIES_KEYS:
        summary_headers.extend([
            f"{short_key}_delta_mean",
            f"{short_key}_delta_last4",
            f"{short_key}_after_last4",
        ])

    summary_path = outdir / "mechanistic_summary.tsv"
    with summary_path.open("w", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=summary_headers, delimiter="\t")
        writer.writeheader()
        for group, group_rows in sorted(groups.items()):
            summary = summarize_group(group_rows)
            record = {"group": group, **summary}
            writer.writerow({key: format_value(record.get(key)) for key in summary_headers})

    print("\t".join(summary_headers))
    for group, group_rows in sorted(groups.items()):
        summary = summarize_group(group_rows)
        record = {"group": group, **summary}
        print("\t".join(format_value(record.get(key)) for key in summary_headers))
    print(f"Wrote {rows_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
