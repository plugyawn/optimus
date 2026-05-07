# Phase8 Systems Report

## Executive Call

- Fastest raw full-search row: `phase8_extra_pod1/search_chunk8_p512_tok16` at `5.772` candidates/sec, but this row is accepted only if its parity gate passes.
- Fastest matched-protocol full search: `phase8_sustain_pod1/search_chunk4_p1024` at `4.791` candidates/sec.
- `max_new_tokens=16` is rejected as a search accelerator: it is faster but fails matched ranking parity.
- Eager mode is rejected: lower throughput and failed top-8 parity.
- `chunk_adapters=4` is the fastest full-search setting; `chunk_adapters=8` is the more conservative reference when strict top-8 parity matters at larger population.
- Staged search with `stage_prompts=8, survivors=64` is the best first-stage triage setting observed: zero selected regret on p64 and p128 panels, with low top-8 recall.

## Full Search

| suite | run | population | screen_prompts | chunk_adapters | max_new_tokens | candidate_sec | screen_prompts_per_sec | eval_elapsed_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phase8_extra_pod1 | search_chunk8_p512_tok16 | 512 | 64 | 8 | 16 | 5.772 | 369.4 | 88.71 |
| phase8_followup_pod1 | search_chunk16_p512_tok16 | 512 | 64 | 16 | 16 | 5.51 | 352.6 | 92.93 |
| phase8_sustain_pod1 | search_chunk4_p1024 | 1024 | 64 | 4 | 32 | 4.791 | 306.6 | 213.7 |
| phase8_extra_pod1 | search_chunk4_p512 | 512 | 64 | 4 | 32 | 4.71 | 301.5 | 108.7 |
| phase8_extra_pod1 | search_chunk8_p1024 | 1024 | 64 | 8 | 32 | 4.357 | 278.8 | 235 |
| phase8_systems_pod1 | search_chunk8_p512 | 512 | 64 | 8 | 32 | 4.355 | 278.7 | 117.6 |
| phase8_extra_pod1 | search_chunk8_p512_mbt16384 | 512 | 64 | 8 | 32 | 4.297 | 275 | 119.2 |
| phase8_extra_pod1 | search_chunk12_p512 | 512 | 64 | 12 | 32 | 4.162 | 266.4 | 123 |
| phase8_systems_pod1 | search_chunk16_p512 | 512 | 64 | 16 | 32 | 4.136 | 264.7 | 123.8 |
| phase8_followup_pod1 | search_chunk16_p512_mbt16384 | 512 | 64 | 16 | 32 | 4.031 | 258 | 127 |
| phase8_systems_pod1 | search_chunk32_p512 | 512 | 64 | 32 | 32 | 3.791 | 242.6 | 135.1 |
| phase8_extra_pod2 | search_chunk8_p512_tok16 | 512 | 128 | 8 | 16 | 3.086 | 395 | 165.9 |
| phase8_extra_pod2 | search_chunk4_p512 | 512 | 128 | 4 | 32 | 2.379 | 304.5 | 215.2 |
| phase8_systems_pod2 | search_chunk8_p512 | 512 | 128 | 8 | 32 | 2.235 | 286.1 | 229.1 |

## Parity Gates

| suite | run | trusted_name | candidate_name | spearman | top8_overlap | selected_regret_vs_trusted | pass |
| --- | --- | --- | --- | --- | --- | --- | --- |
| phase8_extra_pod1 | chunk8_vs_chunk12 | chunk8 | chunk12 | 0.9329 | 5 | 0 | False |
| phase8_extra_pod1 | chunk8_vs_chunk4 | chunk8 | chunk4 | 0.9382 | 6 | 0 | True |
| phase8_extra_pod1 | tok32_vs_chunk8_tok16 | tok32 | tok16 | 0.424 | 2 | 0.1094 | False |
| phase8_extra_pod2 | chunk8_vs_chunk12 | chunk8 | chunk12 | 0.9541 | 5 | 0 | False |
| phase8_extra_pod2 | chunk8_vs_chunk4 | chunk8 | chunk4 | 0.9601 | 4 | 0.007812 | False |
| phase8_extra_pod2 | tok32_vs_chunk8_tok16 | tok32 | tok16 | 0.5181 | 1 | 0.03125 | False |
| phase8_followup_pod1 | compiled_vs_eager | compiled | eager | 0.8894 | 5 | 0 | False |
| phase8_followup_pod1 | tok32_vs_tok16 | tok32 | tok16 | 0.4169 | 2 | 0.01562 | False |
| phase8_sustain_pod1 | chunk8_vs_chunk4_p1024 | chunk8p1024 | chunk4p1024 | 0.9433 | 5 | 0.01562 | False |
| phase8_systems_pod1 | chunk16_vs_chunk32 | chunk16 | chunk32 | 0.9341 | 5 | 0 | False |
| phase8_systems_pod1 | chunk16_vs_chunk8 | chunk16 | chunk8 | 0.9345 | 6 | 0 | True |
| phase8_systems_pod2 | chunk16_vs_chunk32 | chunk16 | chunk32 | 0.9578 | 5 | 0 | False |
| phase8_systems_pod2 | chunk16_vs_chunk8 | chunk16 | chunk8 | 0.9535 | 5 | 0.02344 | False |

## Staged Search

| suite | run | screen_prompts | stage_prompts | survivors | candidate_sec | prompt_eval_savings | top8_survivor_recall | full_best_survived | halving_selected_regret_vs_full |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phase8_extra_pod1 | halving_stage4_surv32_vs_full_chunk8 | 64 | 4 | 32 | 12.32 | 0.875 | 3 | False | 0.03125 |
| phase8_extra_pod1 | halving_stage8_surv64_vs_full_chunk8 | 64 | 8 | 64 | 9.715 | 0.75 | 4 | True | 0 |
| phase8_extra_pod2 | halving_stage4_surv32_vs_full_chunk8 | 128 | 4 | 32 | 10.52 | 0.9062 | 0 | False | 0.01562 |
| phase8_extra_pod2 | halving_stage8_surv64_vs_full_chunk8 | 128 | 8 | 64 | 7.711 | 0.8125 | 2 | True | 0 |

## Gate Summary

- Passing parity rows: `2/13`.
- Zero-regret staged rows with full best survived: `2/4`.

Plots: `full_search_candidate_sec.png`, `parity_gates.png`, `halving_tradeoff.png`.
