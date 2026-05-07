# Update Geometry Audit

- seed: `20260507`
- sigma: `0.01`
- sign: `1`
- sparsity threshold: `1e-05`
- effective-rank energy threshold: `0.99`

| family | total sparsity | total Frobenius norm | mean effective rank | weighted effective-rank fraction |
| --- | ---: | ---: | ---: | ---: |
| dense_gaussian | 0.000956 | 2.213381 | 109.000 | 0.877604 |
| factor_gaussian_lora | 0.000997 | 2.140414 | 8.000 | 0.062500 |
| projected_gaussian_rank_r | 0.002421 | 0.926897 | 8.000 | 0.062500 |

Dense and factor-Gaussian LoRA can be Frobenius-scale matched while having very different rank/correlation geometry.
