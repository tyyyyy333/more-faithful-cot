# V2 First-k Sweep Results

Configuration: Phi-3 mini on ARC-Challenge, `npo_KL`, `lr=5e-4`, LoRA rank 16/alpha 32, two shards, mechanistic diagnostics enabled. All rows below are complete full-sample runs: `547 + 393 = 940` rows over 230 questions.

## Behavior Summary

| k | rows | questions | final flip % | efficacy mean | specificity mean |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 940 | 230 | 13.085 | 86.818 | 99.064 |
| 2 | 940 | 230 | 13.936 | 88.883 | 98.947 |
| 4 | 940 | 230 | 17.128 | 91.545 | 98.995 |
| 8 | 940 | 230 | 19.574 | 95.158 | 99.085 |
| 16 | 940 | 230 | 19.787 | 96.900 | 99.085 |

## Mechanistic Summary

| k | `repr_step_answer_delta_last4` | `repr_step_answer_after_last4` | `attn_answer_to_prefix_delta_last4` | `attn_answer_to_post_step_delta_last4` |
| ---: | ---: | ---: | ---: | ---: |
| 1 | -0.055519 | 0.721338 | -0.011605 | 0.011398 |
| 2 | -0.052321 | 0.724536 | -0.010661 | 0.010434 |
| 4 | -0.045560 | 0.731297 | -0.007387 | 0.006976 |
| 8 | -0.035321 | 0.741537 | -0.003119 | 0.002503 |
| 16 | -0.031232 | 0.745625 | -0.000997 | 0.000433 |

## Interpretation

The first-k sweep supports a front-loaded unlearning effect. Suppressing only the first few target-step tokens already produces a large fraction of the full-step behavioral effect: `k=4` reaches 17.128% final flips, while `k=8` and `k=16` rise to 19.574% and 19.787%. The marginal gain from `k=8` to `k=16` is small, suggesting saturation after the early step prefix.

This is consistent with the "tip-of-the-pen forgetting" hypothesis: the beginning of a reasoning step appears to act like an entry point or trigger for the rest of that reasoning trajectory. It is more precise to say the effect is front-loaded in the target-step prefix than to say the full step knowledge is literally stored in the first tokens.

The mechanistic diagnostics move in the same direction. Smaller `k` causes stronger negative shifts in the late-layer step-answer representation cosine and answer-to-prefix attention mass, while larger `k` gives stronger behavior-level unlearning and higher efficacy. This suggests a tradeoff between highly localized prefix disruption and broader step suppression.

Source summaries:

- `v2_outputs/reproduction/firstk_sweep/summary/v2_result_summary.csv`
- `v2_outputs/reproduction/firstk_sweep/mechanistic/mechanistic_summary.tsv`
