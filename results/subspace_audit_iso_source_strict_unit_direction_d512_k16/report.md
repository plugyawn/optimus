# Subspace Audit

- Rows loaded: `1862`
- Antithetic pairs: `425`
- Sketch dim: `512`
- Splits: `30`
- Split mode: `source`
- Feature scale: `unit`

## Candidate Prediction

| algorithm | spearman_mean | topk_overlap_mean | regret_mean | mean_lift_mean | selected_best_mean |
| --- | --- | --- | --- | --- | --- |
| mean_direction | 0.271296 | 3.43333 | 0.0145833 | 0.0470845 | 0.16875 |
| perm_mean_direction | -0.0135911 | 0.3 | 0.0640625 | 0.00489699 | 0.119271 |
| perm_power_energy_k16 | 0.0953251 | 0.133333 | 0.0734375 | 0.00525506 | 0.109896 |
| perm_ridge | -0.0143119 | 0.133333 | 0.0760417 | -0.00910041 | 0.107292 |
| power_energy_k16 | 0.337344 | 4.73333 | 0.0078125 | 0.0447082 | 0.175521 |
| ridge | 0.339108 | 1.2 | 0.0338542 | 0.00655714 | 0.149479 |

## Antithetic Direction Test

| algorithm | pair_spearman_mean | sign_accuracy_mean | chosen_lift_mean | oracle_gap_mean |
| --- | --- | --- | --- | --- |
| antithetic_mean_gradient | 0.324385 | 0.615319 | 0.00510696 | 0.0137058 |
| perm_antithetic_mean_gradient | 0.010232 | 0.509727 | 0.000441994 | 0.0183707 |

