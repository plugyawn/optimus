# Spectral vLLM Multi-Run Gate

Pass: `false`

## Aggregate

| metric | value |
| --- | ---: |
| runs | 1 |
| parity_pass_count | 0 |
| confirmation_pass_count | 1 |
| validity_pass_count | 1 |
| min_spearman | 0.2998645398025927 |
| mean_spearman | 0.2998645398025927 |
| max_selected_regret | 0.015625 |
| min_ensemble_delta | 0.0 |
| min_full_without_load_speedup | 8.87462881649789 |

## Gates

| gate | pass | detail |
| --- | ---: | --- |
| min_runs | false | `{"min_runs": 2, "runs": 1}` |
| all_validity_pass | true | `{"results/spectral_vllm_confirmation_rank32_c1p5_p16_default": {"control": true, "dense": true, "spectral": true}}` |
| all_quality_parity_pass | false | `{"results/spectral_vllm_confirmation_rank32_c1p5_p16_default": {"gates": {"ensemble_quality": true, "selected_regret": false, "shared_panel": true, "spearman": false, "speed": false, "topk_overlap": true}, "missing_arm": null, "pass": false}}` |
| all_confirmation_pass | true | `{"results/spectral_vllm_confirmation_rank32_c1p5_p16_default": []}` |
| zero_regret_within_k | true | `{"results/spectral_vllm_confirmation_rank32_c1p5_p16_default": 2}` |
| positive_full_speedup | true | `{"min_full_without_load_speedup": 8.87462881649789, "threshold": 1.0}` |
| prompt_robust_selection | false | `{"results/spectral_vllm_confirmation_rank32_c1p5_p16_default": {"min_prompt_variants": 2, "prompt_variant_count": 1, "selection_prompt_variants": ["default"]}}` |

## Runs

| run | parity | confirmation | zero-regret k | full speedup | prompts | spearman | regret | ensemble delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `results/spectral_vllm_confirmation_rank32_c1p5_p16_default` | false | true | 2 | 8.87462881649789 | 1 | 0.2998645398025927 | 0.015625 | 0.0 |
