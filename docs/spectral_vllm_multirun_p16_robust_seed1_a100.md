# Spectral vLLM Robust P16 Seed 1

Run:

```text
results/spectral_vllm_multirun_rank32_c1p5_p16_robust/seed20260507
```

This was launched as the first seed of a two-seed robust multirun. The second
seed was stopped after this seed produced a dense-referenced negative result,
because continuing would spend another full PEFT seed on a configuration already
falsified by the stricter gate.

## Verdict

```text
same-family vLLM -> spectral PEFT confirmation: pass
dense-referenced confirmation: fail
dense parity: fail
validity: pass for dense/control/spectral
```

The systems path is real for recovering the best spectral PEFT candidate, but
it does not recover the dense Gaussian best under the configured shortlist. This
configuration should not be scaled as a dense RandOpt replacement.

## Core Numbers

| arm | base holdout | best ensemble holdout | candidate/sec |
| --- | ---: | ---: | ---: |
| dense Gaussian | 7.03% | 8.59% | 0.0247 |
| factor LoRA control | 7.03% | 7.81% | 0.0208 |
| spectral rank-32 c1.5 | 7.03% | 8.59% | 0.0207 |

Dense parity failed:

| comparison | Spearman | top-8 overlap | selected regret | ensemble delta | speed/dense |
| --- | ---: | ---: | ---: | ---: | ---: |
| factor control | -0.0087 | 4 | 3.125 pp | -0.781 pp | 0.841 |
| spectral | 0.2999 | 7 | 1.5625 pp | 0.0 pp | 0.838 |

The ensemble tie is not enough: the selected spectral screen winner is still
worse under the dense reference ranking, and ranking correlation remains weak.

## Confirmation Split

Same-family confirmation passed:

```text
zero-regret k: 2
best recovered k: 2
eval-only speedup at k=2: 10.98x
full-without-PEFT-load speedup at k=2: 5.35x
```

Dense-referenced confirmation failed:

```text
zero dense-regret k: none
dense best recovered k: 16
dense regret at k=8 confirmed pick: 3.125 pp
dense regret at k=16 confirmed pick: 3.125 pp
```

The dense best candidate was:

```text
dense_gaussian:seed1219141227:s0.002:sign1
```

The spectral/vLLM confirmed candidate was:

```text
spectral_projected_gaussian_rank_r_c1p5:seed591259173:s0.002:sign1
```

Even when the dense best appears in the full k=16 shortlist, the spectral PEFT
confirmation still selects the spectral best, not the dense best. That is the
important failure mode.

## Prompt And Validity

The vLLM proposal used `default` and `reordered` as selection prompt variants;
`xml` was stress-only. Seed 1 validity passed for all PEFT arms. Selected
holdout cap-hit was zero for dense and control, and low for spectral.

The partial second seed has only `vllm_spectral/summary.json` and is not a
quality artifact.
