# Projected Bridge Smoke P16 A100

Date: 2026-05-07

Pod: `fc43be1a2da749338405299236dd8abc`, 1x A100 80GB

Run root: `results/projected_bridge_smoke_p16_a100`

## Setup

- Model: `Qwen/Qwen2.5-3B-Instruct`
- Task: local generated Countdown stress panel, exact-answer reward
- Backend: Transformers/PEFT LoRA and dense in-process mutation
- Population: `P=16`
- Screen prompts: `64`
- Holdout prompts: `256`
- Targets: `q_proj,v_proj`
- Sigma grid: `0.0005,0.001,0.002`
- Generation: `max_new_tokens=128`, `stop_at_answer`
- Promoted candidates: `4`
- Ensembles: `K=1,4`

This is not an upstream-equivalent run. It is a bridge diagnostic for whether
cheap low-rank perturbation families can preserve dense Gaussian RandOpt
selection behavior.

## Families Tested

- `factor_gaussian_lora`: standard independent LoRA factor noise.
- `projected_gaussian_rank_r`: exact best rank-r SVD projection of the same dense Gaussian direction. This is diagnostic, not a fast systems path.
- `randomized_projected_gaussian_rank_r`: randomized range-finder approximation with one power iteration.
- `spectral_projected_gaussian_rank_r`: cheap direct spectral analogue of the leading dense-Gaussian rank-r projection, using Haar-like left/right factors and Gaussian edge-scale singular values.

## Rank Sweep

| rank | arm | pass | Spearman | top-k overlap | regret | speed/dense | mean mutation s | mutation/dense | dense ensemble | arm ensemble | ensemble delta |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | factor LoRA | false | -0.0418 | 4 | 0.0156 | 0.841 | n/a | n/a | 0.0898 | 0.0859 | -0.0039 |
| 8 | exact projected | false | 0.1367 | 4 | 0.0156 | 0.242 | n/a | n/a | 0.0898 | 0.0820 | -0.0078 |
| 32 | factor LoRA | false | 0.1081 | 4 | 0.0469 | 0.845 | 0.074 | 0.050 | 0.0898 | 0.0898 | 0.0000 |
| 32 | exact projected | false | 0.3325 | 6 | 0.0000 | 0.237 | 56.923 | 38.331 | 0.0898 | 0.0898 | 0.0000 |
| 32 | randomized projected | false | 0.4840 | 7 | 0.0156 | 0.673 | 5.040 | 3.394 | 0.0898 | 0.1016 | +0.0117 |
| 32 | spectral projected | false | 0.7355 | 6 | 0.0156 | 0.790 | 0.243 | 0.164 | 0.0898 | 0.0898 | 0.0000 |
| 64 | factor LoRA | false | -0.0781 | 3 | 0.0000 | 0.881 | 0.186 | 0.125 | 0.0898 | 0.0977 | +0.0078 |
| 64 | randomized projected | false | 0.7402 | 6 | 0.0000 | 0.693 | 4.420 | 2.977 | 0.0898 | 0.0898 | 0.0000 |
| 64 | spectral projected | false | 0.3426 | 4 | 0.0000 | 0.801 | 0.496 | 0.334 | 0.0898 | 0.0820 | -0.0078 |

All result-validity audits passed for randomized and spectral selected
candidates: disjoint screen/holdout, strict parser replay, low cap-hit, and low
malformed rates.

## Interpretation

The exact SVD bridge showed that dense-like rank-r geometry can matter:
rank-32 exact projected recovered the dense selected seed with zero regret and
top-8 overlap 6/8. But exact SVD is unusable as a search primitive here:
around 57 seconds of mutation per candidate, roughly 38x dense mutation cost.

The randomized/power-style bridge was informative but not fast enough. Rank-64
randomized projected gave the cleanest dense-parity behavior among the new
methods: Spearman 0.740, top-8 overlap 6/8, zero regret, and equal ensemble
holdout. Its mutation time was still about 4.42 seconds per candidate, around
3x dense mutation cost and around 24x factor-LoRA mutation cost.

The direct spectral approximation did what it was supposed to do on systems
cost: rank-32 mutation dropped to about 0.24 seconds and rank-64 to about
0.50 seconds, much cheaper than randomized projection and exact projection.
Quality did not hold at rank-64. Rank-32 spectral had a promising Spearman
0.735 and top-8 overlap 6/8, but still had selected regret. Rank-64 spectral
regressed to Spearman 0.343, top-8 overlap 4/8, and negative ensemble delta.

The important correction is that plain factor LoRA is not a faithful dense
rank-r projection analogue. The geometry smoke showed factor LoRA is closer to
full dense Frobenius scale, while exact/spectral projected updates have much
lower rank-r projection energy. That explains why factor LoRA can occasionally
get a lucky holdout result while failing ranking parity.

## Decision

Do not scale any of these arms directly.

Preserve these as evidence:

1. Exact projected dense geometry has signal but is too slow.
2. One-iteration randomized projection improves parity at rank 64 but is still too slow.
3. Cheap spectral projection is systemically attractive but not robust enough in this P16 smoke.
4. Factor LoRA remains a useful cheap baseline, not a proven dense Gaussian replacement.

The next high-leverage run is not a larger P with the same hard-coded family.
The code now exposes explicit spectral scale variants:

```text
spectral_projected_gaussian_rank_r_c0p5
spectral_projected_gaussian_rank_r_c0p75
spectral_projected_gaussian_rank_r_c1p25
spectral_projected_gaussian_rank_r_c1p5
spectral_projected_gaussian_rank_r_c2
```

Use those to test whether the spectral family can be corrected with a better
singular-value scale before spending a larger population. Then compare the best
calibrated spectral variant under an accelerated vLLM/SGLang screen where
adapter construction and hotswap costs are measured separately from generation
throughput.
