# Result Validity Audit

Run: `results/spectral_vllm_confirmation_rank32_c1p5_p16_default/dense`
Pass: `true`

| check | pass | detail |
| --- | ---: | --- |
| summary_screen_unique_prompts_matches_total | true | `{"screen_prompts": 64, "screen_unique_prompts": 64}` |
| summary_holdout_unique_prompts_matches_total | true | `{"holdout_prompts": 128, "holdout_unique_prompts": 128}` |
| summary_screen_unique_semantic_prompts_matches_total | true | `{"screen_prompts": 64, "screen_unique_semantic_prompts": 64}` |
| summary_holdout_unique_semantic_prompts_matches_total | true | `{"holdout_prompts": 128, "holdout_unique_semantic_prompts": 128}` |
| summary_screen_holdout_overlap_zero | true | `{"screen_holdout_overlap": 0}` |
| base_screen_rows_present | true | `{"rows": 64}` |
| base_holdout_rows_present | true | `{"rows": 128}` |
| screen_base_ids_unique[default] | true | `{"prompt_variant": "default", "rows": 64, "unique": 64}` |
| screen_base_semantics_unique[default] | true | `{"prompt_variant": "default", "rows": 64, "unique": 64}` |
| holdout_base_ids_unique[default] | true | `{"prompt_variant": "default", "rows": 128, "unique": 128}` |
| holdout_base_semantics_unique[default] | true | `{"prompt_variant": "default", "rows": 128, "unique": 128}` |
| screen_holdout_ids_disjoint | true | `{"overlap": 0}` |
| screen_holdout_semantics_disjoint | true | `{"overlap": 0}` |
| candidate_score_metric_exact_answer | true | `{"candidate_score_metric": "exact_answer"}` |
| ensemble_vote_metric_numeric | true | `{"ensemble_vote_metric": "valid_numeric_majority_vote"}` |
| ensemble_per_prompt_rows_present | true | `{"expected_ks": [1, 4, 8, 16], "present_ks": [1, 4, 8, 16]}` |
| candidate_holdout_rows_present | true | `{"holdout_mode_counts": {"base_holdout": 128, "holdout": 2048}}` |
| stored_rows_match_current_strict_parser | true | `{"mismatches": 0, "sample": []}` |
| selected_candidate_cap_hit_below_threshold | true | `{"by_candidate": {"dense_gaussian:seed1219141227:s0.002:sign1": 0.0, "dense_gaussian:seed1245646949:s0.0005:sign1": 0.0, "dense_gaussian:seed1267011527:s0.002:sign1": 0.0, "dense_gaussian:seed1283028042:s0.002:sign1": 0.0, "dense_gaussian:seed1474485656:s0.002:sign1": 0.0, "dense_gaussian:seed1532350378:s0.002:sign1": 0.0, "dense_gaussian:seed1617523:s0.002:sign1": 0.0, "dense_gaussian:seed1884135719:s0.0005:sign1": 0.0078125, "dense_gaussian:seed1907982355:s0.0005:sign1": 0.0, "dense_gaussian:seed1973132888:s0.001:sign1": 0.0, "dense_gaussian:seed2103302888:s0.0005:sign1": 0.0, "dense_gaussian:seed2113567481:s0.001:sign1": 0.0, "dense_gaussian:seed414161978:s0.0005:sign1": 0.0, "dense_gaussian:seed591259173:s0.002:sign1": 0.0, "dense_gaussian:seed723039624:s0.002:sign1": 0.0, "dense_gaussian:seed765487669:s0.001:sign1": 0.0}, "max": 0.0078125, "threshold": 0.1}` |
| selected_candidate_malformed_below_threshold | true | `{"by_candidate": {"dense_gaussian:seed1219141227:s0.002:sign1": 0.0, "dense_gaussian:seed1245646949:s0.0005:sign1": 0.015625, "dense_gaussian:seed1267011527:s0.002:sign1": 0.0078125, "dense_gaussian:seed1283028042:s0.002:sign1": 0.0078125, "dense_gaussian:seed1474485656:s0.002:sign1": 0.015625, "dense_gaussian:seed1532350378:s0.002:sign1": 0.0234375, "dense_gaussian:seed1617523:s0.002:sign1": 0.0, "dense_gaussian:seed1884135719:s0.0005:sign1": 0.0078125, "dense_gaussian:seed1907982355:s0.0005:sign1": 0.0078125, "dense_gaussian:seed1973132888:s0.001:sign1": 0.0078125, "dense_gaussian:seed2103302888:s0.0005:sign1": 0.0078125, "dense_gaussian:seed2113567481:s0.001:sign1": 0.0078125, "dense_gaussian:seed414161978:s0.0005:sign1": 0.0078125, "dense_gaussian:seed591259173:s0.002:sign1": 0.0, "dense_gaussian:seed723039624:s0.002:sign1": 0.0, "dense_gaussian:seed765487669:s0.001:sign1": 0.015625}, "max": 0.0234375, "threshold": 0.1}` |
| summary_top_holdout_cap_hit_below_threshold | true | `{"max": 0.0078125, "threshold": 0.1}` |
| summary_top_holdout_malformed_below_threshold | true | `{"max": 0.0234375, "threshold": 0.1}` |
