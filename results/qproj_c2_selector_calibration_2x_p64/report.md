# Selector Calibration Audit

Runs: `results/qproj_c2_vllm_shortlist_p64, results/qproj_c2_vllm_shortlist_p64_seed20260508`

## Verdict

Gate: **FAIL**

| metric | value |
| --- | ---: |
| fixed selector heldout pass count | 0/2 |
| linear calibrated heldout pass count | 0/2 |

A pass means the selector chosen or trained on the other panel recovers the dense best within top-8 on the held-out panel.

## Per-Run Selector Diagnostics

### qproj_c2_vllm_shortlist_p64

| selector | Spearman | dense best rank | top-8 contains best | top-8 regret |
| --- | ---: | ---: | --- | ---: |
| valid_min_lift | -0.250013 | 53 | False | 0.015625 |
| mean_minus_malformed | -0.121492 | 53 | False | 0.015625 |
| proposal_exact | -0.0265967 | 59 | False | 0.015625 |
| valid_mean_exact | -0.0265967 | 59 | False | 0.015625 |
| valid_mean_lift | -0.0265967 | 59 | False | 0.015625 |
| default_minus_instability | -0.243272 | 59 | False | 0.015625 |
| current_selection | 0.00296238 | 60 | False | 0.015625 |
| default_exact | 0.140117 | 64 | False | 0.015625 |
| valid_max_lift | 0.141619 | 64 | False | 0.015625 |
| reordered_exact | -0.247238 | 47 | False | 0.046875 |
| low_spread_valid_mean | -0.31738 | 49 | False | 0.046875 |
| xml_exact | -0.155045 | 53 | False | 0.046875 |

### qproj_c2_vllm_shortlist_p64_seed20260508

| selector | Spearman | dense best rank | top-8 contains best | top-8 regret |
| --- | ---: | ---: | --- | ---: |
| default_exact | 0.171311 | 26 | False | 0 |
| valid_min_lift | 0.181133 | 26 | False | 0 |
| default_minus_instability | 0.218444 | 26 | False | 0 |
| low_spread_valid_mean | 0.216819 | 28 | False | 0 |
| mean_minus_malformed | 0.00789465 | 29 | False | 0 |
| current_selection | 0.082039 | 31 | False | 0 |
| proposal_exact | 0.0715169 | 33 | False | 0 |
| valid_mean_exact | 0.0715169 | 33 | False | 0 |
| valid_mean_lift | 0.0715169 | 33 | False | 0 |
| xml_exact | 0.118165 | 13 | False | 0.015625 |
| reordered_exact | -0.038131 | 34 | False | 0.015625 |
| valid_max_lift | -0.0477511 | 34 | False | 0.015625 |

## Held-Out Folds

| train | test | selector | Spearman | dense best rank | top-8 contains best | top-8 regret |
| --- | --- | --- | ---: | ---: | --- | ---: |
| qproj_c2_vllm_shortlist_p64_seed20260508 | qproj_c2_vllm_shortlist_p64 | chosen_fixed:default_exact | 0.140117 | 64 | False | 0.015625 |
| qproj_c2_vllm_shortlist_p64_seed20260508 | qproj_c2_vllm_shortlist_p64 | linear_calibrated | -0.356759 | 28 | False | 0.0625 |
| qproj_c2_vllm_shortlist_p64 | qproj_c2_vllm_shortlist_p64_seed20260508 | chosen_fixed:valid_min_lift | 0.181133 | 26 | False | 0 |
| qproj_c2_vllm_shortlist_p64 | qproj_c2_vllm_shortlist_p64_seed20260508 | linear_calibrated | -0.254985 | 28 | False | 0.015625 |

## Interpretation

This audit is deliberately offline. It does not prove final LoRA quality because only the original robust shortlist has trusted PEFT confirmation. It tests the cheaper prerequisite: whether vLLM-derived scores can be calibrated to rank dense Gaussian screen winners across panels.
