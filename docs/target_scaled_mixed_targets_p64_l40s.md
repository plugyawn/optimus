# Target-Scaled Mixed Targets P64 L40S

## Run

```text
date: 2026-05-08
pod: randopt-target-scale-l40s, Prime pod 54e2fc2e167b438ebaf6b16f6d19e5fb
gpu: 1x NVIDIA L40S 48GB
local base commit before run: a6b064c
model: Qwen/Qwen2.5-3B-Instruct
population: 64
screen prompts: 64
holdout prompts: 128 for PEFT dense/confirmed, 8 for vLLM proposal
rank: 32
sigma values: 0.0005, 0.001, 0.002
prompt variants for vLLM proposal: default, reordered, xml
vLLM: 0.10.2
```

This run tested whether shape-normalized mixed-target LoRA is a higher-leverage
perturbation family than the q-only c2 baseline. The dense PEFT reference was
shared across arms:

```text
dense_gaussian, targets=q_proj
best screen exact: 6/64 = 9.375%
best strict holdout ensemble: 9/128 = 7.031%
candidate/sec: 0.0625
```

The tested mixed-target arms were:

```text
q+v matched-reference:
  activation_spectral_lora_tscale_q2_v1p045

q+v global-budget:
  activation_spectral_lora_tscale_q1p886_v0p985

q+k+v+o global-budget:
  activation_spectral_lora_tscale_q1p333_k0p697_v0p697_o1p333
```

## Result

The systems result is positive, but the quality result is not a go signal.

| arm | vLLM cand/sec | vLLM screen | adapter build | confirmed best screen | best strict holdout | best k | full speedup at best k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| q+v matched-reference | 1.198 | 53.41s | 6.07s | 6/64 = 9.375% | 9/128 = 7.031% | 1 or 8 | 7.77x / 4.66x |
| q+v global-budget | 1.233 | 51.90s | 5.54s | 5/64 = 7.812% | 10/128 = 7.812% | 4 | 6.83x |
| q+k+v+o global-budget | 1.101 | 58.11s | 14.21s | 8/64 = 12.500% | 11/128 = 8.594% | 8 | 5.14x |

The q+k+v+o global-budget arm is the best of these three, but it only matches
the earlier q-only c2 P64 benchmark neighborhood rather than beating it. The
pre-registered go threshold for a strong positive on this seed was at least
13/128 strict holdout. None of the target-scaled mixed-target arms cleared it.

## Validity

All three confirmed PEFT runs passed `result_validity`.

| arm | validity | max cap-hit | max malformed |
| --- | --- | ---: | ---: |
| q+v matched-reference | pass | 0.000% | 5.469% |
| q+v global-budget | pass | 0.000% | 2.344% |
| q+k+v+o global-budget | pass | 0.781% | 2.344% |

This is not a repeat of the earlier token-cap failure mode. The selected
rollouts mostly closed answers, cap-hit was low, and parser rescoring matched
stored rows.

## Selector Alignment

The vLLM shortlist path is fast but still not a reliable dense-screen proxy.

| arm | dense vs vLLM selection Spearman | dense best rank by selection | confirmed vs selection Spearman | dense best recovered in top 8 |
| --- | ---: | ---: | ---: | --- |
| q+v matched-reference | -0.134 | 14 | -0.007 | false |
| q+v global-budget | -0.144 | 28 | 0.122 | false |
| q+k+v+o global-budget | -0.310 | 49 | 0.602 | false |

For q+k+v+o, vLLM did better on the confirmed shortlist than on the full dense
population, but this does not rescue the selector. The dense best still ranked
49/64 by robust selection, and the dense-best recovery gate failed.

## Verdict

```text
Supported:
  vLLM multi-LoRA screening remains much faster than all-candidate PEFT.
  Shape-normalized mixed targets are valid to evaluate.
  q+k+v+o global-budget is the strongest mixed-target arm tested here.

Not supported:
  mixed-target scaling is clearly more powerful than q-only c2.
  target scaling is as powerful as dense Gaussian RandOpt.
  robust vLLM selection can replace trusted PEFT screening.
  this branch deserves P512 scaling.
```

The branch should be treated as negative/neutral. The best next move is not more
targets or larger P; it is to improve the selector/parameterization jointly.
The q+k+v+o result says broader attention-target support is not obviously bad,
but the gain is too small to justify scaling without a stronger search-quality
signal.

## Artifacts

```text
results/tscale_qv_p64_seed20260507
results/tscale_qv_global_p64_seed20260507
results/tscale_qkvo_global_p64_seed20260507
```

The committed artifacts include summaries, per-prompt ledgers, validity reports,
shortlist reports, and alignment audits. Materialized adapter weights and
`family_state.pt` were intentionally not committed.
