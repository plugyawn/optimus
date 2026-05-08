# Target Scale Audit

rank: `32`
reference: `q_proj` at c=`2.0`
reference LoRA/dense Frobenius ratio: `0.5`

| target | shape | effective rank | ratio at reference c | c for reference ratio | c for LoRA/dense=1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| q_proj | 2048x2048 | 32 | 0.5 | 2 | 4 |
| k_proj | 256x2048 | 32 | 0.957107 | 1.04482 | 2.08963 |
| v_proj | 256x2048 | 32 | 0.957107 | 1.04482 | 2.08963 |
| o_proj | 2048x2048 | 32 | 0.5 | 2 | 4 |

Matched-reference family:

```text
activation_spectral_lora_tscale_q2_k1p045_v1p045_o2
```

The current flat activation-spectral rule is shape-dependent. A single c value is not a fair comparison across q/k/v/o when key/value projections have smaller output width.
