# LoRA Family Sweep P16 Default

Run:

```text
results/lora_family_sweep_rank32_p16_probe/default
```

This was a matched PEFT-confirmed quality probe for:

```text
factor_gaussian_lora
sparse_low_rank_lora_d0p25
sparse_low_rank_lora_d0p125
```

It used the same dense Gaussian reference panel, `rank=32`, `P=16`,
`prompts=64`, `holdout_prompts=128`, `sigma_values=0.0005,0.001,0.002`,
`max_new_tokens=128`, and `ensemble_ks=1,4,8`.

The reordered prompt variant was intentionally stopped after the default
variant completed. The default variant already failed the strict family-sweep
gate; continuing would not make this run a two-variant pass artifact.

## Verdict

```text
family sweep gate: fail
factor validity: pass
sparse d0p25 validity: pass
sparse d0p125 validity: pass
dense validity: pass
```

`sparse_low_rank_lora_d0p125` is the only interesting signal: it beat factor
LoRA by 2/128 holdout examples on best ensemble and beat dense by 1/128 on this
single default-prompt panel. But it still failed dense ranking and selected
regret, and it was slower than dense on the PEFT path.

## Core Numbers

| arm | best screen | best ensemble holdout | ensemble delta vs factor | Spearman vs dense | selected regret vs dense | speed/dense |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| factor LoRA | 9.375% | 7.8125% | 0/128 | -0.0087 | 3.125 pp | 0.832 |
| sparse d0p25 | 9.375% | 7.8125% | 0/128 | -0.1098 | 6.25 pp | 0.808 |
| sparse d0p125 | 10.9375% | 9.375% | +2/128 | -0.1273 | 6.25 pp | 0.800 |
| dense Gaussian | 9.375% | 8.59375% | n/a | n/a | n/a | 1.0 |

## Interpretation

The narrow sparse family does make the sampled population luckier on this one
default-prompt panel: d0p125 found a higher screen candidate and a better k=4
ensemble than factor LoRA. That is a research lead, not a pass.

The blockers are:

```text
1. only one prompt variant completed;
2. sparse ranking versus dense is worse than factor;
3. selected dense regret is worse than factor;
4. PEFT candidate/sec is still below dense;
5. d0p25 did not improve over factor.
```

The next useful sparse test should be smaller but prompt-robust, or should move
the sparse d0p125 proposal into the accelerated vLLM shortlist path and confirm
only the top candidates with PEFT. A full PEFT all-arms sweep is too slow for
iteration.
