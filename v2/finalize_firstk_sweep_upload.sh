#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONDA_ENV="${CONDA_ENV:-pf-a100}"
SUMMARY_DIR="${SUMMARY_DIR:-v2_outputs/reproduction/firstk_sweep}"
RESULT_DIR="v2_outputs/final_results/arc-challenge/Phi-3"
REPORT="docs/v2_firstk_sweep_results.md"
K_VALUES="${K_VALUES:-1 2 4 8 16}"
EXPECTED_ROWS="${EXPECTED_ROWS:-940}"
WAIT_PID="${WAIT_PID:-}"

timestamp() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

line_count_for_prefix() {
  local prefix="$1"
  find "$RESULT_DIR" -maxdepth 1 -type f -name "*_${prefix}_s*.out" -print0 \
    | xargs -0 -r wc -l \
    | awk '/ total$/ {print $1; found=1} END {if (!found && NR == 1) print $1; if (NR == 0) print 0}'
}

if [[ -n "$WAIT_PID" ]]; then
  log "Waiting for PID $WAIT_PID before finalizing"
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 60
  done
fi

log "Refreshing first-k summaries"
mkdir -p "$SUMMARY_DIR"
conda run -n "$CONDA_ENV" python v2/summarize_results.py \
  --results-root v2_outputs/final_results \
  --pattern "*test_firstk*_only_diag_s*.out" \
  --outdir "$SUMMARY_DIR/summary"
conda run -n "$CONDA_ENV" python v2/summarize_mechanistic_diagnostics.py \
  --results-dir "$RESULT_DIR" \
  --pattern "*test_firstk*_only_diag_s*.out" \
  --outdir "$SUMMARY_DIR/mechanistic"

complete=1
for k in $K_VALUES; do
  prefix="test_firstk${k}_only_diag"
  rows="$(line_count_for_prefix "$prefix")"
  log "$prefix rows=$rows"
  if [[ "$rows" -lt "$EXPECTED_ROWS" ]]; then
    complete=0
  fi
done

if [[ "$complete" != "1" ]]; then
  log "Not uploading: first-k sweep is incomplete"
  exit 1
fi

log "Writing $REPORT"
tmp_report_script="$(mktemp)"
trap 'rm -f "$tmp_report_script"' EXIT
cat > "$tmp_report_script" <<'PYCODE'
from pathlib import Path
import csv
import re

summary_path = Path("v2_outputs/reproduction/firstk_sweep/mechanistic/mechanistic_summary.tsv")
report_path = Path("docs/v2_firstk_sweep_results.md")

rows = []
with summary_path.open() as infile:
    reader = csv.DictReader(infile, delimiter="\t")
    for row in reader:
        match = re.fullmatch(r"test_firstk(\d+)_only_diag", row["group"])
        if match:
            row["k"] = int(match.group(1))
            rows.append(row)

rows.sort(key=lambda item: item["k"])

def fmt(row, key):
    value = row.get(key, "")
    try:
        return f"{float(value):.4f}"
    except ValueError:
        return value

lines = [
    "# V2 First-k Sweep Results",
    "",
    "Configuration: Phi-3 mini on ARC-Challenge, `npo_KL`, `lr=5e-4`, LoRA rank 16/alpha 32, two shards, mechanistic diagnostics enabled.",
    "",
    "| k | rows | questions | final_flip_pct | efficacy_mean | specificity_mean | repr_step_answer_delta_last4 | attn_answer_to_prefix_delta_last4 |",
    "|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        "| {k} | {rows} | {questions} | {final_flip_pct} | {efficacy_mean} | {specificity_mean} | {repr_delta} | {attn_prefix} |".format(
            k=row["k"],
            rows=row["rows"],
            questions=row["questions"],
            final_flip_pct=fmt(row, "final_flip_pct"),
            efficacy_mean=fmt(row, "efficacy_mean"),
            specificity_mean=fmt(row, "specificity_mean"),
            repr_delta=fmt(row, "repr_step_answer_delta_last4"),
            attn_prefix=fmt(row, "attn_answer_to_prefix_delta_last4"),
        )
    )
lines.extend([
    "",
    "Source summaries:",
    "- `v2_outputs/reproduction/firstk_sweep/summary/v2_result_summary.csv`",
    "- `v2_outputs/reproduction/firstk_sweep/mechanistic/mechanistic_summary.tsv`",
])
report_path.write_text("\n".join(lines) + "\n")
PYCODE
conda run -n "$CONDA_ENV" python "$tmp_report_script"

log "Committing and pushing first-k sweep summaries"
git add v2/run_firstk_sweep_queue.sh v2/run_firstk_sweep_remainder.sh v2/finalize_firstk_sweep_upload.sh "$REPORT" docs/v2_experiment_registry.md
git add -f "$SUMMARY_DIR/summary/v2_result_summary.csv" "$SUMMARY_DIR/mechanistic/mechanistic_summary.tsv"
if git diff --cached --quiet; then
  log "No changes to commit"
else
  git commit -m "Add first-k sweep results"
fi
git push -u origin "$(git branch --show-current)"
log "Upload complete"
