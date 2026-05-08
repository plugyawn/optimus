# Activation Spectral LoRA P32 A100

Date: 2026-05-08

Pod: `d1b05eff5fae4f51aaa01e93f8b1cce7`, 1x A100 80GB

Run roots:

- `results/activation_spectral_gate_p32_a100/activation_c2`
- `results/activation_spectral_calibration_p32_a100/activation_c0p75_fullsigma`
- `results/activation_spectral_calibration_p32_a100/activation_c1_fullsigma`
- `results/activation_spectral_sv_p32_a100/activation_sv_c1_fullsigma`

## Question

Can a task-activation-conditioned LoRA family give us the useful part of dense
Gaussian RandOpt while staying adapter-serveable?

The family fixes the LoRA `A` directions to target-minus-anchor activation
right singular vectors, samples random orthogonal `B` directions, and uses a
spectral edge scale. This is a research-axis proposal, not a dense Gaussian
approximation claim.

## Completed Flat-Scale Results

Dense P32 reference:

```text
run: results/vllm_shortlist_sparse_d0p125_p32_probe/dense
base holdout exact: 7.03125%
dense best screen exact: 9.375%
dense best candidate: seed1219141227:s0.002:sign1 in parity-report spec space
```

Flat activation spectral results:

| arm | validity | best ensemble holdout | top screen | Spearman vs dense | top-8 overlap | selected regret | selected malformed | selected cap |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| c0.75 | pass | 7.031% | 9.375% | -0.186 | 3 | 4.6875 pp | 0.0% | 0.0% |
| c1.0 | pass | 10.938% | 10.938% | -0.087 | 2 | 6.25 pp | 1.5625% | 0.0% |
| c2.0 | fail | 17.188% | 14.0625% | -0.367 | 1 | 4.6875 pp | 25.0% | 1.5625% |
| sv c1.0 | pass | 10.938% | 9.375% | -0.220 | 2 | 3.125 pp | 0.0% | 0.0% |

## Interpretation

The flat-scale family is not a dense-equivalent selector. The c1.0 arm is a
valid local improvement over base, but it actively disagrees with the dense
ranking. The c2.0 arm found a much stronger holdout candidate, but the result is
not trustworthy because the selected/high-holdout candidates have malformed and
cap-hit regressions.

The singular-value-weighted arm did not fix the ranking failure. It tied flat
c1.0 on best ensemble holdout and passed validity, but had weaker screen top
score and worse Spearman. That means preserving the activation singular spectrum
is not sufficient by itself.

This establishes a real tradeoff:

```text
lower scale: valid but weak
middle scale: valid local lift but not dense parity
high scale: strong candidate but invalid rollouts
singular-value weighting: cleaner local lift, still not dense parity
```

## Decision

Do not claim activation-spectral LoRA is as powerful as dense Gaussian RandOpt.
It can produce valid local holdout lift on this P32 panel, but the selector is
not aligned with the dense reference.

The next research move should not be another scalar scale sweep. The useful
branches are:

```text
1. candidate-level validity/drift gating around high-scale c2 candidates;
2. per-layer and q/v-specific rank/scale allocation;
3. prompt-robust activation state construction;
4. vLLM proposal plus PEFT confirmation only after a PEFT-valid family is chosen.
```

## Systems Note

The activation family can now be materialized for vLLM adapters by saving the
same `family_state.pt` used by PEFT and passing it through deterministic adapter
generation. This only enables a proposal-speed test; quality still needs PEFT/HF
confirmation until vLLM output parity passes.
