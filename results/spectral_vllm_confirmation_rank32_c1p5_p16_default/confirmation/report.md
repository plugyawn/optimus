# Confirmation Economics

Two-stage confirmation gate: **PASS**

| metric | value |
| --- | ---: |
| trusted candidates | 16 |
| proposal candidates | 16 |
| trusted full screen seconds | 1169.75 |
| proposal screen seconds | 3.63364 |
| proposal load/build seconds | 87.4834 |
| trusted best candidate | `spectral_projected_gaussian_rank_r_c1p5:seed591259173:s0.002:sign1` |
| trusted best score | 0.09375 |
| best recovered at k | 4 |
| zero-regret k | 2 |

## Gate

| check | pass | detail |
| --- | ---: | --- |
| trusted_best_recovered_within_k | True | `{"best_recovered_k": 4, "max_confirm_k": 8}` |
| zero_regret_within_k | True | `{"max_confirm_k": 8, "max_regret": 0.0, "zero_regret_k": 2}` |
| zero_regret_score_threshold | True | `{"k": 2, "max_regret": 0.0, "regret": 0.0}` |
| eval_only_speedup | True | `{"k": 2, "min_speedup": 1.0, "speedup": 26.390104303120395}` |
| full_without_peft_load_speedup | True | `{"k": 2, "min_speedup": 1.0, "speedup": 8.87462881649789}` |

Failed checks: none

## Top-K Confirmation

| k | contains_trusted_best | trusted_topk_overlap | trusted_topk_possible | confirmed_trusted_score | regret_vs_trusted_best | peft_confirm_s | proposal_plus_confirm_s | eval_only_speedup_vs_trusted_full | full_without_peft_load_speedup_vs_trusted_full |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | False | 0 | 1 | 0.0625 | 0.03125 | 16.6568 | 20.2904 | 57.6505 | 10.8538 |
| 2 | False | 1 | 2 | 0.09375 | 0 | 40.6919 | 44.3255 | 26.3901 | 8.87463 |
| 4 | True | 3 | 4 | 0.09375 | 0 | 97.9059 | 101.54 | 11.5202 | 6.18843 |
| 8 | True | 5 | 8 | 0.09375 | 0 | 202.393 | 206.027 | 5.67767 | 3.98539 |
| 16 | True | 16 | 16 | 0.09375 | 0 | 418.384 | 422.017 | 2.77182 | 2.29589 |

The full-without-PEFT-load estimate includes vLLM load and adapter build time but not a separate PEFT model load.
