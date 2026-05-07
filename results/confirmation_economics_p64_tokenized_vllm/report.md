# Confirmation Economics

Two-stage confirmation gate: **PASS**

| metric | value |
| --- | ---: |
| trusted candidates | 64 |
| proposal candidates | 64 |
| trusted full screen seconds | 403.037 |
| proposal screen seconds | 31.5025 |
| proposal load/build seconds | 170.478 |
| trusted best candidate | `factor_gaussian_lora:seed1411240924:s0.0075:sign1` |
| trusted best score | 0.21875 |
| best recovered at k | 1 |
| zero-regret k | 1 |

## Gate

| check | pass | detail |
| --- | ---: | --- |
| trusted_best_recovered_within_k | True | `{"best_recovered_k": 1, "max_confirm_k": 16}` |
| zero_regret_within_k | True | `{"max_confirm_k": 16, "max_regret": 0.0, "zero_regret_k": 1}` |
| zero_regret_score_threshold | True | `{"k": 1, "max_regret": 0.0, "regret": 0.0}` |
| eval_only_speedup | True | `{"k": 1, "min_speedup": 1.0, "speedup": 10.549169810364441}` |
| full_without_peft_load_speedup | True | `{"k": 1, "min_speedup": 1.0, "speedup": 1.9313351095042466}` |

Failed checks: none

## Top-K Confirmation

| k | contains_trusted_best | trusted_topk_overlap | trusted_topk_possible | confirmed_trusted_score | regret_vs_trusted_best | peft_confirm_s | proposal_plus_confirm_s | eval_only_speedup_vs_trusted_full | full_without_peft_load_speedup_vs_trusted_full |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | True | 1 | 1 | 0.21875 | 0 | 6.70309 | 38.2056 | 10.5492 | 1.93134 |
| 2 | True | 2 | 2 | 0.21875 | 0 | 10.5547 | 42.0572 | 9.58307 | 1.89633 |
| 4 | True | 4 | 4 | 0.21875 | 0 | 24.0668 | 55.5693 | 7.25287 | 1.78298 |
| 8 | True | 6 | 8 | 0.21875 | 0 | 50.8033 | 82.3058 | 4.89683 | 1.5944 |
| 16 | True | 11 | 16 | 0.21875 | 0 | 98.7513 | 130.254 | 3.09424 | 1.34019 |
| 32 | True | 23 | 32 | 0.21875 | 0 | 193.233 | 224.735 | 1.79339 | 1.0198 |

The full-without-PEFT-load estimate includes vLLM load and adapter build time but not a separate PEFT model load.
