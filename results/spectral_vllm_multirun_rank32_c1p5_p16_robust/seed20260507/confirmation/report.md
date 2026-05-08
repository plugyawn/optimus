# Confirmation Economics

Two-stage confirmation gate: **PASS**

| metric | value |
| --- | ---: |
| trusted candidates | 16 |
| proposal candidates | 16 |
| trusted full screen seconds | 773.705 |
| proposal screen seconds | 12.2831 |
| proposal load/build seconds | 74.0697 |
| trusted best candidate | `spectral_projected_gaussian_rank_r_c1p5:seed591259173:s0.002:sign1` |
| trusted best score | 0.09375 |
| best recovered at k | 2 |
| zero-regret k | 2 |

## Gate

| check | pass | detail |
| --- | ---: | --- |
| trusted_best_recovered_within_k | True | `{"best_recovered_k": 2, "max_confirm_k": 8}` |
| zero_regret_within_k | True | `{"max_confirm_k": 8, "max_regret": 0.0, "zero_regret_k": 2}` |
| zero_regret_score_threshold | True | `{"k": 2, "max_regret": 0.0, "regret": 0.0}` |
| eval_only_speedup | True | `{"k": 2, "min_speedup": 1.0, "speedup": 10.982842380090919}` |
| full_without_peft_load_speedup | True | `{"k": 2, "min_speedup": 1.0, "speedup": 5.353753601888897}` |

Failed checks: none

## Top-K Confirmation

| k | contains_trusted_best | trusted_topk_overlap | trusted_topk_possible | confirmed_trusted_score | regret_vs_trusted_best | peft_confirm_s | proposal_plus_confirm_s | eval_only_speedup_vs_trusted_full | full_without_peft_load_speedup_vs_trusted_full |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | False | 0 | 1 | 0.046875 | 0.046875 | 29.1681 | 41.4511 | 18.6655 | 6.69754 |
| 2 | True | 1 | 2 | 0.09375 | 0 | 58.1637 | 70.4467 | 10.9828 | 5.35375 |
| 4 | True | 2 | 4 | 0.09375 | 0 | 102.389 | 114.672 | 6.74713 | 4.09929 |
| 8 | True | 6 | 8 | 0.09375 | 0 | 212.878 | 225.161 | 3.43624 | 2.58565 |
| 16 | True | 16 | 16 | 0.09375 | 0 | 412.956 | 425.239 | 1.81946 | 1.54955 |

The full-without-PEFT-load estimate includes vLLM load and adapter build time but not a separate PEFT model load.
