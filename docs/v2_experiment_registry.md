# v2 Experiment Registry

Last updated: 2026-05-19

## Active Results

Only results in `v2_outputs/final_results` are considered active inputs for
`v2/summarize_results.py`.

Current active experiments:

| Experiment | Status | Reason |
| --- | --- | --- |
| `supp_phi3_arc_downproj_r32_lr5e4_posfix_gs2` | valid | Current trusted down-proj baseline after the stats/position-fix workflow. |
| `supp_phi3_arc_alllinear_lr3e4_postcleanup` | valid | Tests whether all-linear LoRA target scope improves FF-HARD at the planned LR/rank setting. |
| `supp_phi3_arc_alllinear_r32_lr5e4_postcleanup` | valid | Tests all-linear target scope at rank/LR comparable to the strongest down-proj baseline. |

Active files:

- `v2_outputs/final_results/arc-challenge/Phi-3/npo_KL_sentencize_s=True_lr=0.0005_rs=1001_pos=True_ff2=False_supp_phi3_arc_downproj_r32_lr5e4_posfix_gs2_s0.out`
- `v2_outputs/final_results/arc-challenge/Phi-3/npo_KL_sentencize_s=True_lr=0.0005_rs=1001_pos=True_ff2=False_supp_phi3_arc_downproj_r32_lr5e4_posfix_gs2_s1.out`
- `v2_outputs/final_results/arc-challenge/Phi-3/npo_KL_sentencize_s=True_lr=0.0003_rs=1001_pos=True_ff2=False_supp_phi3_arc_alllinear_lr3e4_postcleanup_s0.out`
- `v2_outputs/final_results/arc-challenge/Phi-3/npo_KL_sentencize_s=True_lr=0.0003_rs=1001_pos=True_ff2=False_supp_phi3_arc_alllinear_lr3e4_postcleanup_s1.out`
- `v2_outputs/final_results/arc-challenge/Phi-3/npo_KL_sentencize_s=True_lr=0.0005_rs=1001_pos=True_ff2=False_supp_phi3_arc_alllinear_r32_lr5e4_postcleanup_s0.out`
- `v2_outputs/final_results/arc-challenge/Phi-3/npo_KL_sentencize_s=True_lr=0.0005_rs=1001_pos=True_ff2=False_supp_phi3_arc_alllinear_r32_lr5e4_postcleanup_s1.out`

## Archived Results

Older runs, probes, incomplete starts, and pre-cleanup summary artifacts were
moved out of active paths instead of deleted.

Archive locations:

- `v2_outputs/archive/20260514_pre_cleanup/`
- `logs/archive/20260514_pre_cleanup/`

Archived categories:

- Pre-current-summary runs such as `paper_equiv_phi3_arc_lr1e4`, `supp_phi3_arc_downproj_lr1e4`, and `supp_phi3_arc_downproj_lr3e4`.
- Earlier compare/probe/smoke outputs.
- Empty or aborted starts such as `supp_phi3_arc_alllinear_lr3e4`, `supp_phi3_arc_downproj_r32_lr3e4`, and non-`gs2` `r32_lr5e4_posfix`.
- Previous `v2_outputs/reproduction/summary` files generated before cleanup.
- `core.124350`.

## Summary Command

Use this command after each completed active experiment:

```bash
conda run -n pf-a100 python v2/summarize_results.py --results-root v2_outputs/final_results --outdir v2_outputs/reproduction/summary
```

Expected behavior after cleanup:

- `files_found` should equal the number of active `.out` shard files.
- The CSV should not include archived experiments.

## Metric Definitions

The summary CSV is produced by `v2/summarize_results.py`. Unless stated
otherwise, row-level metrics are computed over completed adapter-step rows. A
row usually corresponds to unlearning one target CoT step for one source
question.

Notation:

- `i`: adapter-step row.
- `q(i)`: source question for row `i`.
- `p_i^0`: answer-option probability vector at epoch 0.
- `p_i^T`: answer-option probability vector at the final epoch.
- `a_i^0 = argmax(p_i^0)`: initial predicted answer.
- `a_i^T = argmax(p_i^T)`: final predicted answer.
- `c_i^0`: target CoT-step sequence probability at epoch 0.
- `c_i^T`: target CoT-step sequence probability at the final epoch.
- `A`: rows where the initial no-CoT prediction equals the initial CoT-conditioned prediction.
- `Q`: unique source questions covered by the rows.

### Coverage Metrics

| Metric | Meaning | Formula / rule |
| --- | --- | --- |
| `rows` | Number of completed adapter-step rows in one `.out` file. | `N = count(rows)` |
| `unique_instances` | Number of unique source questions inside one shard/file. | `|{q(i)}|` |
| `unique_adapter_ids` | Number of distinct adapter jobs represented. | `|{adapter_id_i}|` |
| `agree_rows` | Rows where the initial no-CoT answer and initial CoT-conditioned answer agree. | `|A|` |
| `epoch0_rows` | Rows with epoch-0 baseline evaluation. | Count rows containing `unlearning_results["0"]` |
| `specificity_rows` | Rows with valid specificity evaluation. | Count rows with matching baseline/final `specificity_preds` |
| `new_cot_rows` | Rows with generated post-unlearning CoT. | Count rows with final `new_cot` |
| `final_epoch_max` | Largest epoch key observed in `unlearning_results`. | `max(epoch_keys)` |

When merging shards, do not sum `unique_instances` if shards may overlap. Use
the union:

```text
true_union_questions = | union over shards {q(i)} |
```

### Answer-Flip Metrics

| Metric | Meaning | Formula / rule |
| --- | --- | --- |
| `ff_hard_pct` | Row-level final answer flip rate. This is the strictest FF-HARD style metric in the CSV. | `100 * mean_i[ a_i^T != a_i^0 ]` |
| `answer_changed_pct` | Alias of `ff_hard_pct`. | `100 * mean_i[ a_i^T != a_i^0 ]` |
| `ff_hard_agree_pct` | Row-level final answer flip rate restricted to initially agreeing rows. | `100 * mean_{i in A}[ a_i^T != a_i^0 ]` |
| `question_final_ff_hard_pct` | Question-level final flip rate. A question counts as flipped if any of its rows flips at the final epoch. | `100 * |{q in Q : exists i, q(i)=q and a_i^T != a_i^0}| / |Q|` |
| `question_any_epoch_ff_hard_pct` | Question-level any-epoch flip rate. A question counts as flipped if any row flips at any epoch. | `100 * |{q in Q : exists i,t, q(i)=q and a_i^t != a_i^0}| / |Q|` |
| `agree_question_final_ff_hard_pct` | Question-level final flip rate restricted to questions with initially agreeing rows. | Same as `question_final_ff_hard_pct`, computed over `A` questions |
| `agree_question_any_epoch_ff_hard_pct` | Question-level any-epoch flip rate restricted to initially agreeing rows. | Same as `question_any_epoch_ff_hard_pct`, computed over `A` questions |
| `cot_prediction_changed_pct` | Final answer differs from the original CoT-conditioned answer. | `100 * mean_i[ a_i^T != cot_prediction_i ]` |

Row-level and question-level metrics answer different questions. Row-level
FF-HARD asks how often one unlearned step changes the answer. Question-level
FF-HARD asks how often a source question has at least one step whose unlearning
changes the answer.

### Answer-Probability Metrics

| Metric | Meaning | Formula / rule |
| --- | --- | --- |
| `ff_soft_pct_mean` | Mean probability mass removed from the initial answer, without requiring argmax flip. | `mean_i[ 100 * (p_i^0[a_i^0] - p_i^T[a_i^0]) ]` |
| `ff_soft_pct_median` | Median probability mass removed from the initial answer. | `median_i[ 100 * (p_i^0[a_i^0] - p_i^T[a_i^0]) ]` |
| `ff_soft_agree_pct_mean` | Same as `ff_soft_pct_mean`, restricted to initially agreeing rows. | `mean_{i in A}[ 100 * (p_i^0[a_i^0] - p_i^T[a_i^0]) ]` |
| `answer_margin_drop_mean` | Mean drop in answer margin between top-1 and top-2 normalized answer probabilities. | `mean_i[100 * ((top1(p_i^0)-top2(p_i^0)) - (top1(p_i^T)-top2(p_i^T)))]` |
| `answer_margin_drop_agree_mean` | Same as `answer_margin_drop_mean`, restricted to initially agreeing rows. | Mean over `i in A` |

Positive `ff_soft` means the initial answer became less probable. Positive
margin drop means the answer distribution became less decisive.

### CoT-Suppression Metrics

| Metric | Meaning | Formula / rule |
| --- | --- | --- |
| `efficacy_mean` | Mean relative reduction in target CoT-step probability. Higher means stronger step suppression. | `mean_i[100 * (1 - c_i^T / c_i^0)]` |
| `efficacy_median` | Median relative reduction in target CoT-step probability. | `median_i[100 * (1 - c_i^T / c_i^0)]` |
| `efficacy_p10` / `efficacy_p90` | 10th/90th percentile of target step suppression. | Percentiles of `100 * (1 - c_i^T / c_i^0)` |
| `cot_step_prob_drop_mean` | Same quantity as `efficacy_mean`. | `mean_i[100 * (1 - c_i^T / c_i^0)]` |
| `cot_step_prob_drop_median` | Same quantity as `efficacy_median`. | Median over rows |
| `cot_step_logprob_drop_mean` | Mean log-probability drop for the target CoT step. | `mean_i[log(c_i^0) - log(c_i^T)]` |
| `cot_prob_drop_mean` | Relative reduction in the full original CoT probability. | `mean_i[100 * (1 - full_cot_i^T / full_cot_i^0)]` |
| `cot_logprob_drop_mean` | Log-probability drop for the full original CoT. | `mean_i[log(full_cot_i^0) - log(full_cot_i^T)]` |

High efficacy does not guarantee high FF-HARD. Efficacy measures suppression of
the targeted reasoning text; FF-HARD measures whether the final answer argmax
changes.

### Specificity And Loss Metrics

| Metric | Meaning | Formula / rule |
| --- | --- | --- |
| `specificity_mean` | Agreement between baseline and final predictions on held-out same-task examples. Higher means less collateral damage. | `mean_i[100 * (# unchanged specificity preds / # specificity preds)]` |
| `loss_start_mean` | Mean first recorded training loss. | `mean_i[loss_i^start]` |
| `loss_final_mean` | Mean final recorded training loss. | `mean_i[loss_i^final]` |
| `loss_min_mean` | Mean minimum training loss observed per row. | `mean_i[min_t loss_i^t]` |
| `loss_drop_mean` | Mean loss decrease from start to final. | `mean_i[loss_i^start - loss_i^final]` |
| `loss_drop_median` | Median loss decrease from start to final. | `median_i[loss_i^start - loss_i^final]` |

Loss metrics show whether optimization happened. They should be interpreted
together with efficacy, FF-HARD, and specificity, because lower loss alone does
not imply answer flips.

## Causal_CoT And Mechanistic Ablations

Final full-sample comparison after the Causal_CoT LoRA `pos=True`rerun. These metrics are recomputed from raw JSONL rows, using true question-union aggregation across shards.

| Group | Rows | Questions | Row final FF-HARD | Question final FF-HARD | Question any-epoch FF-HARD | Efficacy | Specificity | Final loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_full` (`supp_phi3_arc_alllinear_r32_lr5e4_postcleanup`) | 940 | 230 | 19.681 | 44.348 | 44.783 | 97.136 | 99.112 | 0.145 |
| `repr_only_full` (`test_repr_only_diag`) | 940 | 230 | 19.681 | 44.348 | 45.217 | 97.166 | 99.069 | 0.146 |
| `firstk4_full` (`test_firstk4_only_diag`) | 940 | 230 | 17.128 | 40.000 | 40.000 | 91.545 | 98.995 | 0.206 |
| `firstk4_repr_full` (`test_firstk4_repr_diag_clean`) | 940 | 230 | 17.128 | 40.000 | 40.000 | 91.573 | 98.979 | 0.209 |
| `causal_cot_full_pos` (`critical_lora_causal_cot_i09_unfiltered_pos`) | 940 | 230 | 19.468 | 40.000 | 40.000 | 96.698 | 98.410 | 0.216 |

Mechanistic comparison for critical-step and full-sample Causal_CoT runs:

| Group | Rows | Questions | Final flip % | Efficacy | Specificity | `attn_answer_to_step_delta_last4` | `attn_answer_to_prefix_delta_last4` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `critical_lora_baseline_i09` | 41 | 29 | 21.951 | 98.397 | 99.268 | +0.000279 | -0.002036 |
| `critical_lora_repr_i09` | 41 | 29 | 19.512 | 98.361 | 99.390 | +0.000317 | -0.002108 |
| `critical_lora_causal_cot_i09` (`pos=False`, not strictly comparable) | 42 | 29 | 14.286 | 96.828 | 98.929 | -0.000354 | -0.015730 |
| `critical_lora_causal_cot_i09_unfiltered_pos` | 940 | 230 | 19.468 | 96.698 | 98.410 | -0.000572 | -0.057266 |

Validation checks before upload:

- No active `v2/unlearn.py` or `run_sharded_unlearn.sh` process.
- GPU idle after completion.
- Full Causal_CoT `pos=True` shard row counts: `547 + 393 = 940`.
- Error-keyword scan over `logs/v2_critical_lora_causal_cot_i09_unfiltered_pos*.log` found no `Traceback`, `ERROR`, OOM, or killed-process markers.
- Summary outputs:
  - `v2_outputs/reproduction/critical_lora_full_pos_summary/v2_result_summary.csv`
  - `v2_outputs/reproduction/critical_lora_full_pos_mechanistic/mechanistic_summary.tsv`
  - `v2_outputs/reproduction/critical_lora_full_pos_mechanistic/mechanistic_rows.csv`

Interpretation:

- `repr_loss` does not improve over the all-linear LoRA baseline.
- `Causal_CoT full pos=True` gives the desired negative internal attention shifts, especially `answer_to_prefix`, but does not improve final answer-flip behavior over baseline and slightly reduces specificity.
- Current best behavior-level run remains `supp_phi3_arc_alllinear_r32_lr5e4_postcleanup`; Causal_CoT is useful as a mechanistic signal but not yet a better unlearning objective.

## Next Experiment Rule

Do not reuse an old `OUTPUT_PREFIX`. Use a new prefix with a clear purpose, and
only launch after deciding the hypothesis.

Completed post-cleanup queue:

| Prefix | Status | Purpose |
| --- | --- |
| `supp_phi3_arc_alllinear_lr3e4_postcleanup` | completed | Test whether expanding LoRA targets beyond `down_proj` increases FF-HARD and whether that damages specificity at the planned all-linear setting. |
| `supp_phi3_arc_alllinear_r32_lr5e4_postcleanup` | completed | Match the current trusted baseline's stronger rank/LR more closely while expanding LoRA targets, so target-scope effects are not confounded by a weaker all-linear setting. |

Post-cleanup comparison, true question-union aggregation:

| Group | Rows | Questions | Row final FF-HARD | Question final FF-HARD | Question any-epoch FF-HARD | Agree-question final FF-HARD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `downproj_r32_lr5e4` | 940 | 230 | 4.681 | 12.609 | 13.043 | 10.096 |
| `alllinear_lr3e4` | 940 | 230 | 11.383 | 26.957 | 26.957 | 24.038 |
| `alllinear_r32_lr5e4` | 940 | 230 | 19.681 | 44.348 | 44.783 | 40.865 |

Reproduce this table with:

```bash
python v2/compare_postcleanup_results.py
```

Interpretation:

- LoRA is effective in this setup: all completed post-cleanup runs strongly suppress the target CoT step while keeping specificity high.
- The low FF-HARD observed in the down-proj baseline is primarily a target-scope limitation, not a summary-code artifact.
- Expanding LoRA targets to all linear layers substantially improves answer flips.
- The strongest all-linear run reaches `44.348` question-level final FF-HARD with `99.112` specificity, so no additional experiment is required for the current target-scope diagnosis.
- A future robustness pass could repeat the strongest configuration with another seed, but that is a replication check rather than a blocker for the current conclusion.

Queue entrypoint:

```bash
setsid nohup bash v2/run_postcleanup_queue.sh > logs/v2_postcleanup_queue.log 2>&1 < /dev/null &
```
