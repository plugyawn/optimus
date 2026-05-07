# Subspace Audit

- Rows loaded: `1862`
- Antithetic pairs: `425`
- Sketch dim: `256`
- Splits: `30`
- Split mode: `source`
- Feature scale: `unit`

## Candidate Prediction

| algorithm | spearman_mean | topk_overlap_mean | regret_mean | mean_lift_mean | selected_best_mean |
| --- | --- | --- | --- | --- | --- |
| mean_direction | 0.215751 | 2.13333 | 0.0302083 | 0.0338032 | 0.153125 |
| perm_mean_direction | -0.00451218 | 0.3 | 0.0619792 | 0.00535272 | 0.121354 |
| perm_power_energy_k8 | 0.0738634 | 0.1 | 0.0760417 | 0.00489699 | 0.107292 |
| perm_ridge | -0.014376 | 0.133333 | 0.08125 | -0.00949103 | 0.102083 |
| power_energy_k8 | 0.290359 | 4.53333 | 0.0104167 | 0.0421691 | 0.172917 |
| ridge | 0.226924 | 1.5 | 0.0244792 | 0.00974725 | 0.158854 |

## Antithetic Direction Test

| algorithm | pair_spearman_mean | sign_accuracy_mean | chosen_lift_mean | oracle_gap_mean |
| --- | --- | --- | --- | --- |
| antithetic_mean_gradient | 0.3096 | 0.627835 | 0.00583611 | 0.0129766 |
| perm_antithetic_mean_gradient | 0.0250933 | 0.513833 | 0.000689854 | 0.0181229 |

