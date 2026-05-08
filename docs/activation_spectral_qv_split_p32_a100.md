# Activation Spectral Q/V Split P32 A100

Date: 2026-05-08

Pod: `d1b05eff5fae4f51aaa01e93f8b1cce7`, 1x A100 80GB

Run root:

```text
results/activation_spectral_qv_split_p32_a100
```

## Question

The activation-spectral family had a sharp tradeoff:

```text
c1.0: valid but weak
c2.0: strong exact-solve signal but malformed/cap fragile in q_proj+v_proj
```

This diagnostic asks whether the signal and the fragility localize to `q_proj`
or `v_proj`, instead of treating `q_proj,v_proj` as one uniform perturbation
budget.

## Completed Arms

| arm | base holdout | best strict ensemble | top screen | validity | note |
| --- | ---: | ---: | ---: | --- | --- |
| `q_proj`, c1 | 7.031% | 7.031% | 7.812% | pass | clean negative |
| `q_proj`, c2 | 7.031% | 9.375% | 12.500% | pass | clean local lift |
| `v_proj`, c2 | 7.031% | 6.250% | 9.375% | fail | malformed/fragile negative |

`q_proj`, c1:

```text
top screen: 7.812%
best strict ensemble holdout: 7.031%
selected cap-hit max: 0.000%
selected malformed max: 0.781%
```

This is a clean negative. Removing `v_proj` killed the useful c1 signal rather
than making it cleaner.

`q_proj`, c2:

```text
top screen: 12.500%
best strict ensemble holdout: 9.375%
selected cap-hit max: 0.000%
selected malformed max: 3.906%
```

Top screen candidates:

| candidate | screen exact | malformed | cap-hit |
| --- | ---: | ---: | ---: |
| `seed1245646949:s0.002` | 12.500% | 0.000% | 0.000% |
| `seed723039624:s0.002` | 10.938% | 1.562% | 0.000% |
| `seed1970200126:s0.002` | 9.375% | 1.562% | 0.000% |
| `seed1827632720:s0.001` | 9.375% | 0.000% | 0.000% |
| `seed271202801:s0.002` | 9.375% | 0.000% | 0.000% |

Holdout behavior was weaker than screen for the top-1 candidate, but the top-4
strict ensemble reached 12/128 correct:

```text
k=1 strict: 8/128 = 6.250%
k=4 strict: 12/128 = 9.375%
k=8 strict: 11/128 = 8.594%
```

This is not yet a scale-up result, but it is a real allocation clue: high-scale
`q_proj` alone preserved a clean ensemble lift, while `q_proj` c1 did not.

`v_proj`, c2:

```text
top screen: 9.375%
best strict ensemble holdout: 6.250%
selected cap-hit max: 1.562%
selected malformed max: 19.531%
validity: fail
```

Top screen candidates:

| candidate | screen exact | malformed | cap-hit |
| --- | ---: | ---: | ---: |
| `seed414161978:s0.001` | 9.375% | 4.688% | 0.000% |
| `seed1532350378:s0.002` | 9.375% | 6.250% | 0.000% |
| `seed1828774463:s0.002` | 9.375% | 3.125% | 1.562% |
| `seed1283028042:s0.0005` | 7.812% | 0.000% | 0.000% |
| `seed1973132888:s0.002` | 7.812% | 14.062% | 0.000% |

Several high-scale `v_proj` rows collapsed on screen:

| candidate | screen exact | malformed | cap-hit |
| --- | ---: | ---: | ---: |
| `seed1884135719:s0.002` | 1.562% | 50.000% | 20.312% |
| `seed1219141227:s0.002` | 1.562% | 12.500% | 1.562% |
| `seed723039624:s0.002` | 0.000% | 3.125% | 1.562% |
| `seed1677712344:s0.002` | 0.000% | high enough to fail qualitatively | nonzero risk |

The top-selected holdout did not recover:

```text
k=1 strict: 3/128 = 2.344%
k=4 strict: 7/128 = 5.469%
k=8 strict: 8/128 = 6.250%
```

The result-validity audit failed on selected malformed rows. In particular,
`seed1973132888:s0.002` had 19.531% malformed on holdout.

## Interpretation

The q/v split changes the working hypothesis:

```text
The c2 signal is not only a q+v malformed artifact.
Some useful high-scale signal survives in q_proj alone.
The v_proj high-scale side is a likely source of malformed/cap fragility.
```

At the same time, the q-only top-1 screen winner did not transfer to holdout.
The useful q-only result is the ensemble, not a single promoted candidate.
That means this family still needs prompt-robust and cap-stable confirmation
before any systems claim.

`v_proj` c2 was not merely weaker; it was also invalid by the repo's rollout
validity gate. That makes a uniform `q_proj,v_proj` c2 scale suspect: the old
q+v c2 result may be mixing a useful q-side perturbation with v-side formatting
damage.

## Decision

Do not return to generalized activation bases. The next parameterization work
should be allocation, not fancier basis construction:

```text
1. run prompt-robust confirmation for q_proj c2 before any scale-up;
2. test q_proj c1.5/c2 with lower malformed thresholds and cap stability;
3. test q_proj c2 plus low-scale v_proj only as an ablation, not as default;
4. run vLLM proposal only after the PEFT family passes prompt robustness.
```

The immediate next scientific question is not "more LoRA family variants"; it is
whether q-only high-scale survives prompt variants and cap changes.
