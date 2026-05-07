# Proposal Audit

Scale gate: **FAIL**

Compared `subspace_power` against `random_iso` using saved run artifacts.

| Metric | subspace_power | random_iso | delta |
| --- | ---: | ---: | ---: |
| candidate/sec | 1.55631 | 1.58676 | -0.0304487 |
| screen candidate/sec | 1.73511 | 1.77129 | -0.0361799 |
| best ensemble holdout exact | 0.191406 | 0.189453 | 0.00195312 |
| screen valid fraction | 0.0917969 | 0.0820312 | 0.00976562 |
| screen top16 selection mean | 0.00537109 | 0.0107422 | -0.00537109 |
| screen top16 exact mean | 0.120117 | 0.125 | -0.00488281 |

## Scale Gate

| Check | pass | detail |
| --- | ---: | --- |
| candidate_throughput_not_slower | False | `{"left": 1.5563107298948171, "min_ratio": 1.0, "ratio": 0.9808107469527648, "right": 1.5867594586723752}` |
| screen_throughput_not_slower | False | `{"left": 1.735114146858158, "min_ratio": 1.0, "ratio": 0.9795742908049699, "right": 1.771294084731766}` |
| ensemble_quality_not_worse | True | `{"delta": 0.001953125, "left": 0.19140625, "min_delta": 0.0, "right": 0.189453125}` |
| screen_top16_quality_not_worse | False | `{"delta": -0.0048828125, "left": 0.1201171875, "min_delta": 0.0, "right": 0.125}` |
| screen_valid_fraction_not_worse | True | `{"delta": 0.009765625, "left": 0.091796875, "min_delta": 0.0, "right": 0.08203125}` |
| prompt_selection_rank_stable | False | `{"left": 0.20642369523078127, "min_spearman": 0.5}` |
| prompt_selection_rank_not_worse_than_control | False | `{"delta": -0.029615582560169995, "left": 0.20642369523078127, "min_delta": 0.0, "right": 0.23603927779095127}` |
| screen_to_holdout_transfer_not_worse_than_control | False | `{"delta": -0.23665693503858243, "left": 0.2976582399304869, "min_delta": 0.0, "right": 0.5343151749690693}` |
| screen_selected_regret_not_worse_than_control | True | `{"delta": -0.01171875, "left": 0.00390625, "max_increase": 0.0, "right": 0.015625}` |

Failed checks: candidate_throughput_not_slower, screen_throughput_not_slower, screen_top16_quality_not_worse, prompt_selection_rank_stable, prompt_selection_rank_not_worse_than_control, screen_to_holdout_transfer_not_worse_than_control

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
