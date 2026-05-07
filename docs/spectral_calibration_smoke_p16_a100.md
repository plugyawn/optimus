# Spectral Calibration Smoke P16 A100

Date: 2026-05-07

Pod: `ceb4aaf31313498e9bb505221adb04a1`, 1x A100 80GB

Run roots:

- `results/spectral_calibration_rank32_p16_a100`
- `results/spectral_calibration_rank64_p16_a100`

## Question

The previous projected-bridge smoke showed that direct spectral projected
Gaussian LoRA was much cheaper than exact or randomized projection, but the
hard-coded scale was not quality-robust. This run tested whether a scalar
calibration of the spectral edge singular values fixes the rank-32 selected
regret and rank-64 regression.

This is still a small `P=16` diagnostic, not a scale run.

## Setup

- Model: `Qwen/Qwen2.5-3B-Instruct`
- Task: local generated Countdown stress panel, exact-answer reward
- Backend: Transformers/PEFT LoRA, reusing the prior dense/factor baselines
- Population: `P=16`
- Screen prompts: `64`
- Holdout prompts: `256`
- Targets: `q_proj,v_proj`
- Sigma grid: `0.0005,0.001,0.002`
- Generation: `max_new_tokens=128`, `stop_at_answer`
- Promoted candidates: `4`
- Ensembles: `K=1,4`

## Rank 32 Calibration

| arm | Spearman | top-8 overlap | regret | speed/dense | mean mutation s | mutation/dense | ensemble delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| factor LoRA | 0.108 | 4 | 0.0469 | 0.845 | 0.074 | 0.050 | 0.0000 |
| spectral c0.5 | 0.625 | 5 | 0.0000 | 0.843 | 0.229 | 0.154 | +0.0117 |
| spectral c0.75 | 0.311 | 6 | 0.0000 | 0.836 | 0.203 | 0.137 | -0.0117 |
| spectral c1.25 | 0.567 | 6 | 0.0156 | 0.850 | 0.195 | 0.132 | +0.0039 |
| spectral c1.5 | 0.702 | 6 | 0.0000 | 0.869 | 0.265 | 0.178 | +0.0117 |
| spectral c2 | 0.540 | 5 | 0.0000 | 0.851 | 0.182 | 0.123 | 0.0000 |

The best rank-32 tradeoff was `spectral_projected_gaussian_rank_r_c1p5`:
zero selected regret, top-8 overlap 6/8, Spearman 0.702, and +1.17pp ensemble
holdout over dense/base. It still failed strict parity because Spearman and
total candidate/sec did not pass the hard gates.

`c0p5` also produced +1.17pp ensemble holdout, but with weaker top-8 overlap.

## Rank 64 Follow-Up

Only the two rank-32 candidates worth checking were promoted to rank 64:
`c0p5` as the lower-scale ensemble control and `c1p5` as the best rank-32
tradeoff.

| arm | Spearman | top-8 overlap | regret | speed/dense | mean mutation s | mutation/dense | ensemble delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| factor LoRA | -0.078 | 3 | 0.0000 | 0.881 | 0.186 | 0.125 | +0.0078 |
| spectral c0.5 | 0.176 | 3 | 0.0000 | 0.844 | 0.496 | 0.334 | 0.0000 |
| spectral c1.5 | 0.523 | 5 | 0.0000 | 0.843 | 0.433 | 0.291 | -0.0078 |

Rank 64 did not confirm the rank-32 signal. `c1p5` improved the ranking shape
relative to factor LoRA, but lost ensemble quality. `c0p5` preserved ensemble
quality but had weak ranking parity.

## Decision

Do not scale calibrated spectral projection directly.

The useful evidence is narrower:

1. Spectral scaling is a real lever at rank 32.
2. The rank-32 gain is not stable across rank.
3. Mutation cost is low enough to keep spectral families in the systems
   candidate set.
4. Current spectral families do not meet the full objective: they are not
   faster in total PEFT candidate/sec and do not have robust dense-Gaussian
   parity.

The next productive direction should change the system path, not only the
spectral scale. A calibrated spectral family should be tested inside the
accelerated vLLM/SGLang adapter screen, where model reload and PEFT mutation
overhead are removed. If it still fails there, the remaining direction is not
scalar spectral calibration; it is either a different structured noise family
or a reuse/speculative systems strategy across many adapters.

