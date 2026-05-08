# RandOpt LoRA Goal Audit

Pass: `false`

| requirement | pass | evidence | detail |
| --- | ---: | --- | --- |
| official full-Gaussian baseline validity | false | `missing` | `"missing evidence"` |
| quality parity | false | `results/spectral_vllm_confirmation_rank32_c1p5_p16_default/parity/summary.json` | `{"ensemble_delta": 0.0, "ensemble_quality": true, "missing_arm": null, "overall_pass": false, "selected_arm": "lora"}` |
| stability parity | false | `results/spectral_vllm_confirmation_rank32_c1p5_p16_default/parity/summary.json` | `{"gates": {"selected_regret": false, "spearman": false, "topk_overlap": true}, "missing_arm": null, "selected_arm": "lora", "selected_regret": 0.015625, "spearman": 0.2998645398025927, "topk_overlap": 7}` |
| speed parity | false | `results/spectral_vllm_confirmation_rank32_c1p5_p16_default/parity/summary.json` | `{"gate": false, "missing_arm": null, "selected_arm": "lora", "speed_ratio_lora_over_dense": 0.8329453944048366}` |
| trusted accelerated backend selector | false | `missing` | `"missing evidence"` |
| two-stage accelerated confirmation | true | `results/spectral_vllm_confirmation_rank32_c1p5_p16_default/confirmation/summary.json` | `{"best_recovered_k": 4, "failed": [], "pass": true, "thresholds": {"max_confirm_k": 8, "max_regret": 0.0, "min_eval_only_speedup": 1.0, "min_full_without_load_speedup": 1.0}, "zero_regret_k": 2}` |
| multi-run prompt-robust confirmation | false | `results/spectral_vllm_multirun_gate_p16_default/summary.json` | `{"aggregate": {"confirmation_pass_count": 1, "max_selected_regret": 0.015625, "mean_spearman": 0.2998645398025927, "min_ensemble_delta": 0.0, "min_full_without_load_speedup": 8.87462881649789, "min_spearman": 0.2998645398025927, "parity_pass_count": 0, "runs": 1, "validity_pass_count": 1}, "failed": ["min_runs", "all_quality_parity_pass", "prompt_robust_selection"], "pass": false, "thresholds": {"max_zero_regret_k": 8, "min_full_without_load_speedup": 1.0, "min_prompt_variants": 2, "min_runs": 2, "parity_arm": "lora"}}` |
| prompt robustness | false | `missing` | `"missing prompt robustness report"` |
| drift parity | false | `missing` | `"missing drift evidence"` |
| eval validity | true | `results/spectral_vllm_confirmation_rank32_c1p5_p16_default/spectral/validity/summary.json` | `[]` |
| adapter convenience | true | `results/spectral_vllm_confirmation_rank32_c1p5_p16_default/vllm_spectral` | `{"adapters_jsonl_exists": true, "summary_adapters_kept": true}` |
