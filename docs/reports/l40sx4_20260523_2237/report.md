# Optimus Systems Report

## Executive Call

- Fastest raw full-search row: `optimus_gpu_suite_v092_noflash_tp4/search_p4096_chunk8` at `3.335` candidates/sec.
- Fastest matched-protocol full search: `optimus_gpu_suite_v092_noflash_tp4/search_p4096_chunk8` at `3.335` candidates/sec.
- Quality rows separate screen-selected heldout transfer from promoted holdout-oracle quality.
- Staged search, when present, is judged by selected regret, full-best survival, and top-k survivor recall on matched full-search panels.

## Full Search

| suite | run | population | screen_prompts | chunk_adapters | tensor_parallel_size | max_new_tokens | candidate_sec | screen_prompts_per_sec | screen_tokens_per_sec | best_tokens_per_sec | eval_elapsed_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| optimus_gpu_suite_v092_noflash_tp4 | search_p4096_chunk8 | 4096 | 64 | 8 | 4 | 32 | 3.335 | 224.7 | 3980 | 3980 | 1228 |
| optimus_gpu_suite_v092_noflash_tp4 | search_p1024_chunk8 | 1024 | 64 | 8 | 4 | 32 | 3.103 | 241.3 | 4298 | 4298 | 330 |

## Adapter Throughput

| suite | run | adapters | prompts | tensor_parallel_size | lora_tokens_per_sec | mixed_tokens_per_sec | lora_prompts_per_sec | mixed_prompts_per_sec | load_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| optimus_gpu_suite_v092_noflash_tp4 | bench_a8_p64 | 8 | 64 | 4 | None | 2718 | None | 146 | 138.8 |

## Quality Scaling

| suite | run | population | base_holdout_exact | screen_selected_holdout_exact | screen_selected_holdout_delta_vs_base | promoted_holdout_oracle_exact | promoted_holdout_oracle_delta_vs_base | best_ensemble_holdout_exact | best_strict_ensemble_holdout_exact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| optimus_gpu_suite_v092_noflash_tp4 | search_p1024_chunk8 | 1024 | 0.07031 | 0.1016 | 0.03125 | 0.1484 | 0.07812 | None | None |
| optimus_gpu_suite_v092_noflash_tp4 | search_p4096_chunk8 | 4096 | 0.07031 | 0.1094 | 0.03906 | 0.1484 | 0.07812 | None | None |

## Parity Gates

| suite | run | trusted_name | candidate_name | spearman | top8_overlap | selected_regret_vs_trusted | pass |
| --- | --- | --- | --- | --- | --- | --- | --- |

## Staged Search

| suite | run | screen_prompts | stage_prompts | survivors | candidate_sec | prompt_eval_savings | top8_survivor_recall | full_best_survived | halving_selected_regret_vs_full |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Gate Summary

- Passing parity rows: `0/0`.
- Zero-regret staged rows with full best survived: `0/0`.
- Best-of-N points: `5120`.

Plots: `full_search_candidate_sec.png`, `token_throughput.png`, `adapter_throughput.png`, `best_of_n.png`, `quality_scaling.png`.
