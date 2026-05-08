# QProj C2 vLLM Shortlist P64 Seed 20260508 A100

## Run

```text
date: 2026-05-08
pod: qproj-c2-replica-p64, Prime pod ac503c7b7d8648059f7649c5ee13e245
gpu: 1x NVIDIA A100-SXM4-80GB
pod status after sync: terminated, prime pods list empty
out: results/qproj_c2_vllm_shortlist_p64_seed20260508
model: Qwen/Qwen2.5-3B-Instruct
dense reference: dense_gaussian, targets=q_proj
proposal family: activation_spectral_lora_c2, targets=q_proj
population: 64
screen prompts: 64
holdout prompts: 128 for PEFT dense/confirmed, 8 for vLLM proposal
rank: 32
sigma values: 0.0005, 0.001, 0.002
prompt variants for vLLM proposal: default, reordered, xml
vLLM: 0.20.1
torch: 2.11.0+cu130
```

This is the fresh-seed replication of the first q-only c2 P64 vLLM shortlist
test. The data split and candidate seed were changed to `20260508`.

## Systems Result

The proposal path remains much faster than a trusted all-candidate PEFT screen:

```text
dense PEFT full screen: 962.46 s
dense PEFT candidate/sec: 0.0665
vLLM load + adapter build: 92.95 s
vLLM proposal screen: 48.31 s
vLLM proposal candidate/sec: 1.3247
screen-only throughput ratio: 19.92x
```

The end-to-end speedup after PEFT confirmation is still useful:

| k | confirmed strict holdout | dense strict holdout at k | delta vs dense best strict | full speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 3.1250% | 3.9062% | -2.3438 pp | 6.21x |
| 4 | 7.0312% | 4.6875% | +1.5625 pp | 4.92x |
| 8 | 7.8125% | 5.4688% | +2.3438 pp | 4.01x |

The operational search-quality gate passed:

```text
out: results/qproj_c2_vllm_shortlist_p64_seed20260508/search_quality_confirmation
gate: PASS
passing k: 4
dense validity: pass
confirmed validity: pass
```

This is the second P64 panel where `vLLM proposal + PEFT confirmation` beats the
dense strict-holdout ensemble while staying faster.

## Quality Result

The dense seed/spec parity gate failed again:

```text
out: results/qproj_c2_vllm_shortlist_p64_seed20260508/shortlist_dense_confirmation
gate: FAIL
dense best: dense_gaussian:seed931683830:s0.001:sign1
dense best screen score: 9.375%
dense best recovered k: None
zero dense-regret k: None
```

| k | confirmed best | confirmed screen | dense score at same seed/sigma | dense regret | dense best in top-k |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | `activation_spectral_lora_c2:seed1920188830:s0.002:sign1` | 6.2500% | 6.2500% | 3.1250 pp | false |
| 2 | `activation_spectral_lora_c2:seed1920188830:s0.002:sign1` | 6.2500% | 6.2500% | 3.1250 pp | false |
| 4 | `activation_spectral_lora_c2:seed320056936:s0.0005:sign1` | 6.2500% | 7.8125% | 1.5625 pp | false |
| 8 | `activation_spectral_lora_c2:seed1019436086:s0.0005:sign1` | 7.8125% | 7.8125% | 1.5625 pp | false |

So the result should not be described as dense Gaussian parity. The clean claim
is narrower: the q-only c2 LoRA family can be proposed cheaply and PEFT-confirmed
to equal or beat dense strict holdout on this panel, but the current vLLM
selector does not recover the dense winner.

## Alignment Audit

The alignment problem repeated, though it was less extreme than the previous
seed:

| comparison | common | Spearman | mean abs delta |
| --- | ---: | ---: | ---: |
| dense exact vs vLLM selection score | 64 | 0.082 | 0.0834 |
| dense exact vs vLLM exact mean | 64 | 0.0715 | 0.0167 |
| dense exact vs vLLM default exact | 64 | 0.171 | 0.0168 |
| dense exact vs vLLM reordered exact | 64 | -0.038 | 0.0225 |

Dense-best recall under the vLLM selection score:

| k | contains dense best | dense top-k overlap |
| ---: | --- | ---: |
| 1 | false | 0 |
| 2 | false | 0 |
| 4 | false | 0 |
| 8 | false | 1 |
| 16 | false | 5 |
| 32 | true | 17 |

Default and reordered vLLM prompt rankings also disagree:

```text
default-vs-reordered Spearman: 0.161 exact, 0.139 selection
top-4 overlap: 0/4
top-8 overlap: 1/8
```

On the eight PEFT-confirmed shortlisted candidates, vLLM was still only weakly
aligned with trusted PEFT:

| comparison | common | Spearman | mean abs delta |
| --- | ---: | ---: | ---: |
| PEFT exact vs vLLM selection score | 8 | 0.234 | 0.0264 |
| PEFT exact vs vLLM proposal exact | 8 | 0.234 | 0.0283 |
| PEFT exact vs vLLM default exact | 8 | 0.0897 | 0.0313 |
| PEFT exact vs vLLM reordered exact | 8 | 0.000 | 0.0293 |

Default-prompt backend agreement was high in label space but not enough for
ranking:

```text
common default rows: 512
exact-label agreement: 95.3125%
text equality: 60.5469%
vLLM exact mean: 8.3984%
PEFT exact mean: 5.2734%
```

## Rollout Probe

The high single-digit scores are not caused by cap saturation or malformed
parser artifacts:

```text
base holdout: 6/128 exact, malformed 0/128, cap-hit 0/128
confirmed selected max malformed: 1/128 per candidate
confirmed selected max cap-hit: 0/128 per candidate
```

The correct rows are real, short arithmetic equations. Examples where the
confirmed q-only c2 shortlist added correct holdout answers over base:

| example | numbers | target | base answer | corrected answer |
| ---: | --- | ---: | --- | --- |
| 125 | 74, 24, 82, 83 | 97 | `82+24-74` | `74+24+82-83` |
| 263 | 19, 58, 12, 59 | 148 | `19+58+12+59-148` | `19+58+12+59` |
| 384 | 7, 6, 33, 70 | 104 | `70+33-7*6` | `70+33-6+7` |
| 643 | 24, 59, 60, 74 | 99 | `60+24-59` | `60+24-59+74` |

The improvement is still small in absolute terms. The confirmed strict top-8
ensemble got only 10/128 holdout prompts correct, versus 7/128 for dense top-8
and 6/128 for base.

## Two-Seed Interpretation

Across the two q-only c2 P64 panels:

```text
repeated positive:
  vLLM proposal + PEFT confirmation passes the operational strict-holdout gate.
  full wall-clock speedup is roughly 4x-6x at k<=8.
  rollout validity is clean enough to trust the aggregate counts.

repeated negative:
  vLLM selection does not recover dense best within k=8.
  dense-vs-vLLM rank correlation is weak.
  prompt-variant rankings are unstable enough to corrupt robust_mean selection.
```

This supports continuing the accelerated-cascade line, but not scaling the
current selector as-is.

## Next Tests

The next experiments should be smaller and diagnostic, not a larger population:

1. Calibrate the selector on the two P64 panels.
   Use vLLM features such as default/reordered/xml exacts, per-variant rank,
   variance, malformed rate, cap-hit rate, output length, sigma, and seed spec.
   Gate on held-out panel dense-best recall and PEFT-confirmed strict holdout,
   not in-panel fit.

2. Stop treating `robust_mean(default,reordered)` as authoritative.
   The two prompt variants are useful stress tests, but their rankings disagree
   too much for direct averaging to be a safe selector.

3. Test dimension-normalized target allocation.
   Same c across `q_proj` and `v_proj` is not shape-fair. Try q c2 plus smaller
   v/k/o scales before concluding q-only is intrinsically best.

4. Sweep q-only rank and scale separately.
   Test rank 16, 32, 64 with both fixed c and roughly fixed Frobenius norm.
   This distinguishes "rank helps" from "larger perturbation norm helps".

5. Keep PEFT confirmation as the authority.
   vLLM is currently a fast proposal engine, not a trusted scorer.
