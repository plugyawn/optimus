# Dense-Referenced Confirmation

Gate: **FAIL**

| metric | value |
| --- | ---: |
| shared candidates | 16 |
| dense best | `dense_gaussian:seed1219141227:s0.002:sign1` |
| dense best score | 0.09375 |
| spectral best | `spectral_projected_gaussian_rank_r_c1p5:seed591259173:s0.002:sign1` |
| spectral best score | 0.09375 |
| zero dense-regret k | None |
| dense best recovered k | 16 |

## Gate

| check | pass | detail |
| --- | ---: | --- |
| zero_dense_regret_within_k | false | `{"max_confirm_k": 8, "zero_dense_regret_k": null}` |
| dense_regret_threshold | false | `{"dense_regret": null, "k": null, "max_dense_regret": 0.0}` |
| positive_full_speedup_vs_dense | false | `{"k": null, "min_speedup": 1.0, "speedup": null}` |

## Top-K

| k | confirmed spec | dense score | dense regret | spectral score | spectral regret | dense top-k overlap | full speedup |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `seed765487669:s0.001:sign1` | 0.046875 | 0.046875 | 0.046875 | 0.046875 | 0/1 | 5.6136 |
| 2 | `seed591259173:s0.002:sign1` | 0.0625 | 0.03125 | 0.09375 | 0 | 0/2 | 4.4873 |
| 4 | `seed591259173:s0.002:sign1` | 0.0625 | 0.03125 | 0.09375 | 0 | 1/4 | 3.43586 |
| 8 | `seed591259173:s0.002:sign1` | 0.0625 | 0.03125 | 0.09375 | 0 | 6/8 | 2.16719 |
| 16 | `seed591259173:s0.002:sign1` | 0.0625 | 0.03125 | 0.09375 | 0 | 16/16 | 1.29877 |

This is a dense-reference screen-score gate. It does not replace matched holdout or drift parity.
