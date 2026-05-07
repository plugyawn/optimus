# Factor-LoRA Gaussian Parity Rank Sweep

Remote: Prime A100-SXM4 80GB, Qwen/Qwen2.5-3B-Instruct, Countdown exact-answer reward.

Command shape:

```text
population: 64
screen prompts: 64
holdout prompts: 256
sigma: 0.01
targets: q_proj,v_proj
max_new_tokens: 32
stop_at_answer: true
families: dense_gaussian vs factor_gaussian_lora
ranks: 8, 32
```

Dense Gaussian is independent of LoRA rank, so rank 32 reused the rank 8 dense panel.

| rank | Spearman vs dense | top-8 overlap | selected regret | dense cand/s | LoRA cand/s | speed ratio | pass |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 8 | -0.071053 | 1 | 0.09375 | 0.212113 | 0.201436 | 0.949662 | false |
| 32 | -0.018490 | 2 | 0.09375 | 0.212113 | 0.197649 | 0.931810 | false |

Selected-candidate validity:

| rank | LoRA-selected seed | screen exact | dense score at LoRA pick | cap-hit | malformed |
| ---: | --- | ---: | ---: | ---: | ---: |
| 8 | seed1109140247 | 0.140625 | 0.0 | 0.015625 | 0.015625 |
| 32 | seed1838555499 | 0.15625 | 0.0 | 0.0625 | 0.03125 |

Dense best:

```text
seed410114451
screen exact: 0.09375
cap-hit: 0.0
malformed: 0.046875
```

Interpretation:

```text
factor_gaussian_lora rank 8: failed parity
factor_gaussian_lora rank 32: failed parity
```

Both ranks select candidates that score zero under the dense panel, have near-zero
rank correlation with dense Gaussian scores, and are slower than dense in this
trusted HF reference backend. This does not rule out vLLM/SGLang systems wins,
projected-rank bridges, sparse/structured families, or rank/drift retuning, but
it does kill the claim that the current factor-Gaussian LoRA family is already
Gaussian-RandOpt-equivalent at rank 8 or rank 32.

