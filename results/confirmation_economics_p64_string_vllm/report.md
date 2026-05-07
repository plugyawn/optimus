# Confirmation Economics

| metric | value |
| --- | ---: |
| trusted candidates | 64 |
| proposal candidates | 64 |
| trusted full screen seconds | 403.037 |
| proposal screen seconds | 18.0527 |
| proposal load/build seconds | 87.5173 |
| trusted best candidate | `factor_gaussian_lora:seed1411240924:s0.0075:sign1` |
| trusted best score | 0.21875 |
| best recovered at k | 1 |
| zero-regret k | 1 |

## Top-K Confirmation

| k | contains_trusted_best | trusted_topk_overlap | trusted_topk_possible | confirmed_trusted_score | regret_vs_trusted_best | peft_confirm_s | proposal_plus_confirm_s | eval_only_speedup_vs_trusted_full | full_without_peft_load_speedup_vs_trusted_full |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | True | 1 | 1 | 0.21875 | 0 | 6.70309 | 24.7558 | 16.2805 | 3.58979 |
| 2 | True | 1 | 2 | 0.21875 | 0 | 13.3886 | 31.4413 | 12.8187 | 3.38804 |
| 4 | True | 4 | 4 | 0.21875 | 0 | 24.0668 | 42.1195 | 9.56889 | 3.10897 |
| 8 | True | 7 | 8 | 0.21875 | 0 | 48.5879 | 66.6406 | 6.04792 | 2.61444 |
| 16 | True | 10 | 16 | 0.21875 | 0 | 98.9188 | 116.971 | 3.4456 | 1.97095 |
| 32 | True | 24 | 32 | 0.21875 | 0 | 196.622 | 214.674 | 1.87743 | 1.33371 |

The full-without-PEFT-load estimate includes vLLM load and adapter build time but not a separate PEFT model load.
