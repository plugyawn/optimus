# Activation Projected And Generalized P32 A100

Date: 2026-05-08

Pod: `d1b05eff5fae4f51aaa01e93f8b1cce7`, 1x A100 80GB

Run roots:

- `results/activation_projected_p32_a100/projected_c2_fullsigma`
- `results/activation_generalized_projected_p32_a100/generalized_c2_fullsigma`
- `results/activation_generalized_spectral_p32_a100/generalized_spectral_c2_fullsigma`

## Question

After the activation-spectral c2 run found a real but malformed-prone candidate,
we tested whether a more dense-faithful activation basis could keep the signal
while reducing rollout fragility.

Two follow-up families were added:

```text
activation_projected_gaussian_rank_r_c2
activation_generalized_projected_gaussian_rank_r_c2
activation_generalized_spectral_lora_c2
```

The projected Gaussian family keeps the dense seed identity by drawing the same
dense Gaussian `G` and projecting it into the activation right basis:

```text
delta = G V V^T
```

The generalized families replace the plain target-activation basis with a
target-vs-anchor generalized eigenspace. The intent was to find directions that
are high-variance on target prompts and quiet on anchor prompts.

## Math Review Fixes

The math review found and the code now fixes these issues:

```text
1. activation collection was including pad rows; it now masks by attention_mask;
2. vllm_lora_halving was missing --prompt-input; it is now threaded through;
3. --activation-state-no-anchor-subtract was silently ineffective for generalized families;
4. generalized spectral variants now exist as norm-controlled alternatives.
```

The corrected generalized implementation should be treated as the tested object
for these smoke runs.

## Results

Dense and activation-spectral references:

| arm | status | key result |
| --- | --- | --- |
| dense Gaussian reference | reference | base holdout 7.031%, dense best screen 9.375% |
| activation spectral c1.0 | valid but weak | best ensemble holdout 10.938%, top screen 10.938% |
| activation spectral c2.0 | invalid method claim | best ensemble holdout 17.188%, but selected malformed 25.0% |

Projected c2 completed:

| metric | value |
| --- | ---: |
| population | 32 |
| screen prompts | 64 |
| holdout prompts | 128 |
| base screen exact | 6.250% |
| base holdout exact | 7.031% |
| best strict ensemble holdout | 8.594% |
| best individual holdout | 9.375% |
| top screen | 10.938% |
| selected cap-hit | 0.000% |

Projected c2 was clean, but the effect size was small. The strict ensemble gain
over base holdout was only +1.5625 percentage points on 128 examples, below the
pre-registered scale-up bar.

Generalized projected c2 was stopped after 15 candidates because the screen
signal was weaker than projected c2:

| candidate | screen exact | malformed | cap-hit |
| --- | ---: | ---: | ---: |
| `seed591259173:s0.002` | 7.812% | 0.000% | 0.000% |
| `seed1267011527:s0.0005` | 7.812% | 0.000% | 0.000% |
| base | 6.250% | 0.000% | 0.000% |

Generalized spectral c2 was stopped after 8 candidates because it was also weak
and high-sigma rows were fragile:

| candidate | screen exact | malformed | cap-hit |
| --- | ---: | ---: | ---: |
| `seed1907982355:s0.0005` | 7.812% | 1.562% | 0.000% |
| `seed414161978:s0.001` | 7.812% | 3.125% | 0.000% |
| base | 6.250% | 0.000% | 0.000% |
| `seed1532350378:s0.002` | 3.125% | 21.875% | 4.688% |
| `seed1973132888:s0.002` | 1.562% | 23.438% | 1.562% |
| `seed591259173:s0.002` | 0.000% | 7.812% | 0.000% |

## Interpretation

The seed-preserving activation-projected family answers the narrow systems
question cleanly: adapter-compatible projected directions can be evaluated with
low candidate-switch overhead, and the run is not dominated by malformed or
cap-hit artifacts. It does not answer the quality question strongly enough. A
strict ensemble of 8.594% versus 7.031% base is a small local lift, not evidence
that the LoRA family is as powerful as dense Gaussian RandOpt.

The generalized target-vs-anchor basis did not help in this smoke. It lowered
the early screen ceiling and, in the spectral form, reintroduced the same
high-scale malformed/cap fragility that made activation-spectral c2 invalid as
a family-level method claim.

This is negative evidence against "make the activation basis fancier" as the
next high-leverage direction. The current best signal remains candidate-level
high-scale activation-spectral behavior, but it needs validity gating or a
better allocation of rank and scale before it can be scaled.

## Decision

Do not scale the generalized activation-basis line.

Preserve it as:

```text
activation_projected_gaussian_rank_r_c2: clean but weak
activation_generalized_projected_gaussian_rank_r_c2: negative partial smoke
activation_generalized_spectral_lora_c2: negative partial smoke plus fragility
```

The next research moves should be:

```text
1. per-layer and q/v-specific rank/scale allocation around the original activation-spectral family;
2. candidate-level validity/drift gating for high-scale c2 candidates;
3. prompt-robust confirmation on any selected candidate before systems claims;
4. vLLM token-ID proposal plus PEFT/HF shortlist confirmation only after quality passes.
```

The next systems move should still avoid speculative decoding as the main
investment. The blocker is not decoder throughput yet; it is finding a
serveable perturbation family whose selected candidates survive strict,
prompt-robust PEFT confirmation.
