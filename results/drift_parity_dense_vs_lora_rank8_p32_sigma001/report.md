# Drift Parity Report

Pass: `true`

| metric | reference | candidate | comparison |
| --- | ---: | ---: | ---: |
| family | dense_gaussian | factor_gaussian_lora |  |
| rows | 32 | 32 |  |
| prompts | 32 | 32 |  |
| KL(base||candidate) mean | 0.00162154 | 0.00139563 | ratio 0.860679 |
| logit L2 mean | 185.597 | 176.894 | ratio 0.953105 |
| top-1 equal mean | 1 | 1 | delta 0 |

## Checks

| check | pass | detail |
| --- | ---: | --- |
| reference_is_logit_drift | True | `{"kind": "logit_drift", "path": "results/drift_dense_p32_sigma001"}` |
| candidate_is_logit_drift | True | `{"kind": "logit_drift", "path": "results/drift_lora_rank8_p32_sigma001"}` |
| rows_present | True | `{"candidate_rows": 32, "min_rows": 32, "reference_rows": 32}` |
| same_prompt_count | True | `{"candidate_prompts": 32, "reference_prompts": 32}` |
| kl_nonnegative | True | `{"candidate_min_kl": 0.00025536061730235815, "reference_min_kl": 0.00021594252029899508}` |
| candidate_mean_kl_not_higher | True | `{"candidate": 0.0013956254288132186, "max_ratio": 1.1, "ratio": 0.860679338609803, "reference": 0.0016215393657148525}` |
| candidate_logit_l2_not_higher | True | `{"candidate": 176.89380264282227, "max_ratio": 1.1, "ratio": 0.9531045998235278, "reference": 185.5974702835083}` |
| candidate_top1_not_worse | True | `{"candidate": 1.0, "delta": 0.0, "min_delta": -0.01, "reference": 1.0}` |

Failed checks: none
