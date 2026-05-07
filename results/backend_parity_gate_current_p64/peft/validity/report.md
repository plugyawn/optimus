# Result Validity Audit

Run: `results/backend_parity_gate_current_p64/peft`
Pass: `false`

| check | pass | detail |
| --- | ---: | --- |
| summary_screen_unique_prompts_matches_total | true | `{"screen_prompts": 64, "screen_unique_prompts": 64}` |
| summary_holdout_unique_prompts_matches_total | true | `{"holdout_prompts": 8, "holdout_unique_prompts": 8}` |
| summary_screen_unique_semantic_prompts_matches_total | true | `{"screen_prompts": 64, "screen_unique_semantic_prompts": 64}` |
| summary_holdout_unique_semantic_prompts_matches_total | true | `{"holdout_prompts": 8, "holdout_unique_semantic_prompts": 8}` |
| summary_screen_holdout_overlap_zero | true | `{"screen_holdout_overlap": 0}` |
| base_screen_rows_present | true | `{"rows": 64}` |
| base_holdout_rows_present | true | `{"rows": 8}` |
| screen_base_ids_unique[default] | true | `{"prompt_variant": "default", "rows": 64, "unique": 64}` |
| screen_base_semantics_unique[default] | true | `{"prompt_variant": "default", "rows": 64, "unique": 64}` |
| holdout_base_ids_unique[default] | true | `{"prompt_variant": "default", "rows": 8, "unique": 8}` |
| holdout_base_semantics_unique[default] | true | `{"prompt_variant": "default", "rows": 8, "unique": 8}` |
| screen_holdout_ids_disjoint | true | `{"overlap": 0}` |
| screen_holdout_semantics_disjoint | true | `{"overlap": 0}` |
| candidate_score_metric_exact_answer | true | `{"candidate_score_metric": "exact_answer"}` |
| candidate_holdout_rows_present | false | `{"holdout_mode_counts": {"base_holdout": 8}}` |
| stored_rows_match_current_strict_parser | true | `{"mismatches": 0, "sample": []}` |
| selected_candidate_cap_hit_below_threshold | true | `{"by_candidate": {}, "max": 0.0, "threshold": 0.1}` |
| selected_candidate_malformed_below_threshold | true | `{"by_candidate": {}, "max": 0.0, "threshold": 0.1}` |
