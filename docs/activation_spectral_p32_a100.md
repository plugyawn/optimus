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

## Strict Ensemble Replay

After adding a strict replay path that rejects rows failing `score_completion`
before numeric voting, the ensemble numbers did not collapse:

| arm | numeric best | strict best |
| --- | ---: | ---: |
| c0.75 | 7.031% | 7.031% |
| c1.0 | 10.938% | 10.938% |
| c2.0 | 17.188% | 17.188% |
| sv c1.0 | 10.938% | 10.938% |

That means the main c2 caveat is not only the lax ensemble vote. There is real
exact-solve signal in the candidate, but the candidate still fails the clean
method bar because individual rollouts are too often malformed or cap-affected.

## C2 Cap/Prompt Audit

Partial audit root:

```text
results/cap_stability_activation_c2_top4
```

The audit was stopped after 4096 saved rows because the failure mode was already
clear and the run was competing with the next A100 smoke. Completed conditions
cover all screen caps/prompts, holdout default caps 64/128, holdout reordered
cap 64, and holdout reordered base/elite_0 at cap 128.

Key rows:

| split | prompt | cap | candidate | exact | malformed | cap-hit |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| screen | default | 64 | base | 6.250% | 0.000% | 0.000% |
| screen | default | 64 | elite_0 | 14.062% | 25.000% | 1.562% |
| screen | reordered | 64 | base | 9.375% | 0.000% | 0.000% |
| screen | reordered | 64 | elite_0 | 9.375% | 0.000% | 0.000% |
| screen | reordered | 64 | elite_1 | 1.562% | 64.062% | 3.125% |
| holdout | default | 64 | base | 7.031% | 0.000% | 0.000% |
| holdout | default | 64 | elite_0 | 17.188% | 13.281% | 0.781% |
| holdout | reordered | 64 | base | 5.469% | 0.000% | 0.000% |
| holdout | reordered | 64 | elite_0 | 14.844% | 0.000% | 0.000% |
| holdout | reordered | 64 | elite_1 | 2.344% | 54.688% | 1.562% |

Takeaways:

```text
1. Raising the cap from 64 to 128/256 did not fix malformed rows.
2. elite_0 has real holdout lift, including reordered holdout lift, but default-prompt malformed is too high.
3. elite_1 and elite_2 show that the high-scale region is unstable: clean/default or high/default screen scores can become severe malformed collapse under reordered prompts.
4. C2 remains an interesting candidate-level phenomenon, not a valid family-level method claim.
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

The immediate follow-up is the seed-preserving
`activation_projected_gaussian_rank_r` family. It samples the same dense
Gaussian seed and projects it into the activation right basis, instead of using
a random left basis. That is a cleaner dense-bridge test than another
activation-spectral scale sweep.

## Systems Note

The activation family can now be materialized for vLLM adapters by saving the
same `family_state.pt` used by PEFT and passing it through deterministic adapter
generation. This only enables a proposal-speed test; quality still needs PEFT/HF
confirmation until vLLM output parity passes.
