# Selector Union Audit

Runs: `results/qproj_c2_vllm_shortlist_p64, results/qproj_c2_vllm_shortlist_p64_seed20260508, results/tscale_qv_p64_seed20260507, results/tscale_qv_global_p64_seed20260507, results/tscale_qkvo_global_p64_seed20260507`

## Verdict

| metric | value |
| --- | ---: |
| run count | 5 |
| first all-run dense-best policy | None |
| first all-run zero-regret policy | None |
| first all-run <=1/64 screen-regret policy | {'policy': 'default_exact', 'k': 4, 'max_dense_regret': 0.015625, 'mean_dense_regret': 0.0125, 'dense_best_recall': 1} |
| pass at k<=8 | false |
| pass at k<=16 | false |

## Aggregate Policy Recall

| policy | k | dense-best recall | mean regret | max regret | mean top-k overlap |
| --- | ---: | ---: | ---: | ---: | ---: |
| all_builtin_rr | 4 | 0/5 | 0.021875 | 0.046875 | 0.2 |
| all_builtin_rr | 8 | 1/5 | 0.0125 | 0.015625 | 1.6 |
| all_builtin_rr | 16 | 2/5 | 0.00625 | 0.015625 | 4.8 |
| all_builtin_rr | 32 | 3/5 | 0.003125 | 0.015625 | 17.4 |
| current_selection | 4 | 0/5 | 0.028125 | 0.046875 | 0.2 |
| current_selection | 8 | 0/5 | 0.021875 | 0.03125 | 0.6 |
| current_selection | 16 | 1/5 | 0.015625 | 0.03125 | 3.8 |
| current_selection | 32 | 3/5 | 0.00625 | 0.015625 | 15 |
| default_exact | 4 | 1/5 | 0.0125 | 0.015625 | 0.8 |
| default_exact | 8 | 1/5 | 0.009375 | 0.015625 | 1.8 |
| default_exact | 16 | 1/5 | 0.009375 | 0.015625 | 4.8 |
| default_exact | 32 | 4/5 | 0.003125 | 0.015625 | 16.8 |
| prompt_exact_rr | 4 | 0/5 | 0.015625 | 0.015625 | 0.4 |
| prompt_exact_rr | 8 | 1/5 | 0.0125 | 0.015625 | 2 |
| prompt_exact_rr | 16 | 3/5 | 0.003125 | 0.015625 | 4.8 |
| prompt_exact_rr | 32 | 4/5 | 0.003125 | 0.015625 | 17 |
| prompt_lift_rr | 4 | 0/5 | 0.015625 | 0.015625 | 0.4 |
| prompt_lift_rr | 8 | 1/5 | 0.0125 | 0.015625 | 1.8 |
| prompt_lift_rr | 16 | 3/5 | 0.003125 | 0.015625 | 4.8 |
| prompt_lift_rr | 32 | 4/5 | 0.003125 | 0.015625 | 17 |
| proposal_exact | 4 | 0/5 | 0.021875 | 0.046875 | 0.2 |
| proposal_exact | 8 | 0/5 | 0.0125 | 0.015625 | 1.4 |
| proposal_exact | 16 | 0/5 | 0.0125 | 0.015625 | 4.6 |
| proposal_exact | 32 | 1/5 | 0.009375 | 0.015625 | 16 |
| stability_rr | 4 | 0/5 | 0.021875 | 0.046875 | 0.4 |
| stability_rr | 8 | 0/5 | 0.015625 | 0.03125 | 1.2 |
| stability_rr | 16 | 3/5 | 0.003125 | 0.015625 | 5 |
| stability_rr | 32 | 3/5 | 0.003125 | 0.015625 | 18.2 |

## Per-Run Dense-Best Rank

| run | policy | dense best rank | k=8 contains best | k=16 contains best |
| --- | --- | ---: | --- | --- |
| qproj_c2_vllm_shortlist_p64 | all_builtin_rr | 64 | false | false |
| qproj_c2_vllm_shortlist_p64 | current_selection | 60 | false | false |
| qproj_c2_vllm_shortlist_p64 | default_exact | 64 | false | false |
| qproj_c2_vllm_shortlist_p64 | prompt_exact_rr | 64 | false | false |
| qproj_c2_vllm_shortlist_p64 | prompt_lift_rr | 64 | false | false |
| qproj_c2_vllm_shortlist_p64 | proposal_exact | 59 | false | false |
| qproj_c2_vllm_shortlist_p64 | stability_rr | 60 | false | false |
| qproj_c2_vllm_shortlist_p64_seed20260508 | all_builtin_rr | 39 | false | false |
| qproj_c2_vllm_shortlist_p64_seed20260508 | current_selection | 31 | false | false |
| qproj_c2_vllm_shortlist_p64_seed20260508 | default_exact | 26 | false | false |
| qproj_c2_vllm_shortlist_p64_seed20260508 | prompt_exact_rr | 30 | false | false |
| qproj_c2_vllm_shortlist_p64_seed20260508 | prompt_lift_rr | 30 | false | false |
| qproj_c2_vllm_shortlist_p64_seed20260508 | proposal_exact | 33 | false | false |
| qproj_c2_vllm_shortlist_p64_seed20260508 | stability_rr | 50 | false | false |
| tscale_qv_p64_seed20260507 | all_builtin_rr | 14 | false | true |
| tscale_qv_p64_seed20260507 | current_selection | 14 | false | true |
| tscale_qv_p64_seed20260507 | default_exact | 17 | false | false |
| tscale_qv_p64_seed20260507 | prompt_exact_rr | 10 | false | true |
| tscale_qv_p64_seed20260507 | prompt_lift_rr | 10 | false | true |
| tscale_qv_p64_seed20260507 | proposal_exact | 23 | false | false |
| tscale_qv_p64_seed20260507 | stability_rr | 13 | false | true |
| tscale_qv_global_p64_seed20260507 | all_builtin_rr | 21 | false | false |
| tscale_qv_global_p64_seed20260507 | current_selection | 28 | false | false |
| tscale_qv_global_p64_seed20260507 | default_exact | 25 | false | false |
| tscale_qv_global_p64_seed20260507 | prompt_exact_rr | 16 | false | true |
| tscale_qv_global_p64_seed20260507 | prompt_lift_rr | 16 | false | true |
| tscale_qv_global_p64_seed20260507 | proposal_exact | 33 | false | false |
| tscale_qv_global_p64_seed20260507 | stability_rr | 14 | false | true |
| tscale_qkvo_global_p64_seed20260507 | all_builtin_rr | 8 | true | true |
| tscale_qkvo_global_p64_seed20260507 | current_selection | 49 | false | false |
| tscale_qkvo_global_p64_seed20260507 | default_exact | 3 | true | true |
| tscale_qkvo_global_p64_seed20260507 | prompt_exact_rr | 6 | true | true |
| tscale_qkvo_global_p64_seed20260507 | prompt_lift_rr | 6 | true | true |
| tscale_qkvo_global_p64_seed20260507 | proposal_exact | 43 | false | false |
| tscale_qkvo_global_p64_seed20260507 | stability_rr | 16 | false | true |

## Interpretation

This audit does not replace PEFT confirmation. It tests whether a cheap vLLM shortlist policy has enough dense-winner recall to justify spending PEFT confirmation budget on its selected candidates.
