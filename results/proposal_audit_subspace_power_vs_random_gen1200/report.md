# Proposal Audit

Compared `subspace_power` against `random_iso` using saved run artifacts.

| Metric | subspace_power | random_iso | delta |
| --- | ---: | ---: | ---: |
| candidate/sec | 1.55631 | 1.58676 | -0.0304487 |
| screen candidate/sec | 1.73511 | 1.77129 | -0.0361799 |
| best ensemble holdout exact | 0.191406 | 0.189453 | 0.00195312 |
| screen valid fraction | 0.0917969 | 0.0820312 | 0.00976562 |
| screen top16 selection mean | 0.00537109 | 0.0107422 | -0.00537109 |
| screen top16 exact mean | 0.120117 | 0.125 | -0.00488281 |

## Prompt Variant Stability

| Metric | subspace_power | random_iso |
| --- | ---: | ---: |
| selection_spearman | 0.206424 | 0.236039 |
| exact_spearman | 0.138702 | 0.183712 |
| selection_mean_abs_delta | 0.661407 | 0.605438 |
| exact_mean_abs_delta | 0.0324707 | 0.0339966 |
| top16_selection_overlap | 1 | 0 |
| common_candidates | 512 | 512 |

## Screen To Holdout Transfer

| Metric | subspace_power | random_iso |
| --- | ---: | ---: |
| common_candidates | 16 | 16 |
| screen_selection_vs_holdout_exact_spearman | 0.297658 | 0.534315 |
| screen_exact_vs_holdout_exact_spearman | 0.810103 | 0.56072 |
| screen_selected_regret | 0.00390625 | 0.015625 |
| screen_selected_holdout_exact | 0.126953 | 0.113281 |
| best_holdout_exact | 0.130859 | 0.128906 |

## Antithetic Pair Structure

| Metric | subspace_power | random_iso |
| --- | ---: | ---: |
| pairs | 231 | 256 |
| pair_best_selection_top_mean | 0.00390625 | 0.0107422 |
| pair_best_exact_top_mean | 0.112793 | 0.125 |
| pair_score_gap_mean | 0.622159 | 0.622223 |
| pair_one_valid_one_invalid_fraction | 0.190476 | 0.15625 |
| pair_all_valid_fraction | 0 | 0.00390625 |
