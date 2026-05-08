# QProj C2 vLLM Shortlist P64 A100

## Run

```text
date: 2026-05-08
pod: qproj-c2-prompt-confirm, Prime pod 62c6857728c64acfb3b22eac3882ef53
gpu: 1x NVIDIA A100-SXM4-80GB
out: results/qproj_c2_vllm_shortlist_p64
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
```

The first attempt failed before vLLM serving because the pod image did not have
`vllm` installed. The dense reference had already completed and passed
`result_validity`, so the rerun installed vLLM, preserved the dense reference,
and reran only vLLM proposal, shortlist extraction, PEFT confirmation, and the
dense-regret report.

## Systems Result

The accelerated path is fast once vLLM is available:

```text
dense PEFT full screen: 1068.44 s
dense PEFT candidate/sec: 0.0599
vLLM proposal screen: 48.25 s
vLLM load + adapter build: 98.06 s
vLLM proposal candidate/sec: 1.3264
```

At the shortlist sizes tested, proposal plus PEFT confirmation remained faster
than the full dense PEFT reference:

| k | proposal + confirm | proposal + load/build + confirm | eval-only speedup | full speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 62.23 s | 160.29 s | 17.17x | 6.67x |
| 2 | 76.25 s | 174.30 s | 14.01x | 6.13x |
| 4 | 89.10 s | 187.16 s | 11.99x | 5.71x |
| 8 | 102.79 s | 200.84 s | 10.39x | 5.32x |

This is a real systems positive for `vLLM proposal + PEFT confirmation`.

## Quality Result

There are two quality questions, and they give different answers.

The dense seed/spec parity gate failed. vLLM did not shortlist the dense best
candidate:

```text
dense best: dense_gaussian:seed1267011527:s0.002:sign1
dense best screen score: 10.9375%
dense best recovered k: None
zero dense-regret k: None
gate: FAIL
```

| k | confirmed best | confirmed screen | dense score at same seed/sigma | dense regret | dense best in top-k |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | `activation_spectral_lora_c2:seed273426313:s0.002:sign1` | 7.8125% | 9.3750% | 1.5625 pp | false |
| 2 | `activation_spectral_lora_c2:seed273426313:s0.002:sign1` | 7.8125% | 9.3750% | 1.5625 pp | false |
| 4 | `activation_spectral_lora_c2:seed273426313:s0.002:sign1` | 7.8125% | 9.3750% | 1.5625 pp | false |
| 8 | `activation_spectral_lora_c2:seed273426313:s0.002:sign1` | 7.8125% | 9.3750% | 1.5625 pp | false |

The PEFT-confirmed q-only c2 shortlist did have a useful strict-ensemble
holdout row:

```text
dense strict holdout:
  k=1: 6.25%
  k=4: 7.03125%
  k=8: 7.03125%

confirmed q-only c2 strict holdout:
  k=1: 6.25%
  k=4: 8.59375%
  k=8: 7.03125%
```

So the family is not a dead end. The failure is narrower but important:

```text
vLLM robust shortlist did not recover the dense best or zero-regret candidate.
q-only c2 is not yet a quality-preserving accelerated replacement for dense
Gaussian RandOpt.
```

However, the operational search-quality gate passes on trusted strict holdout:

```text
out: results/qproj_c2_vllm_shortlist_p64/search_quality_confirmation
gate: PASS
passing k: 4
```

| k | confirmed strict holdout | dense strict holdout at k | delta vs dense best strict | full speedup | pass |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 6.2500% | 6.2500% | -0.781 pp | 6.67x | false |
| 4 | 8.5938% | 7.0312% | +1.5625 pp | 5.71x | true |
| 8 | 7.0312% | 7.0312% | 0.000 pp | 5.32x | true |

This distinction matters. The run does not prove q-only c2 is a faithful
low-rank surrogate for dense Gaussian perturbations, but it does show that this
accelerated LoRA search path can recover an equal-or-better trusted strict
holdout ensemble on this P64 panel with a 5.3x-5.7x full speedup.

## Validity

Both PEFT paths passed `result_validity`.

```text
dense validity: pass
confirmed q-only c2 validity: pass
confirmed selected max cap-hit: 0.78125%
confirmed selected max malformed: 3.90625%
```

The vLLM proposal used `token_ids` input and robust prompt selection over
`default,reordered,xml`, but the prompt-robust vLLM score surface was not
aligned enough with the trusted PEFT screen. The top vLLM proposal score was
only 1.5625% under the robust selection score, while PEFT confirmed several of
those shortlisted candidates at 7.8125% screen exact.

## Alignment Audit

After the run, `randopt_lora_lab.shortlist_alignment_audit` was added to split
the failure into dense-family mismatch, prompt-variant mismatch, and backend
drift. The audit result is:

```text
out: results/qproj_c2_vllm_shortlist_p64/shortlist_alignment_audit
```

The dense-vs-vLLM proposal scores do not agree over the full P64 panel:

| comparison | common | Spearman | mean abs delta |
| --- | ---: | ---: | ---: |
| dense exact vs vLLM selection score | 64 | 0.003 | 0.0886 |
| dense exact vs vLLM exact mean | 64 | -0.027 | 0.0264 |
| dense exact vs vLLM default exact | 64 | 0.140 | 0.0146 |
| dense exact vs vLLM reordered exact | 64 | -0.247 | 0.0464 |

This is not a small top-k accident. Dense's best candidate spec was ranked
`60/64` by vLLM robust selection and `64/64` by vLLM default exact:

```text
dense best spec: seed1267011527:s0.002:sign1
dense score: 10.9375%
vLLM selection rank: 60
vLLM default-exact rank: 64
```

On the eight PEFT-confirmed shortlisted candidates, default-prompt vLLM was
only weakly aligned with PEFT, and the reordered prompt was anti-aligned:

| comparison | common | Spearman | mean abs delta |
| --- | ---: | ---: | ---: |
| PEFT exact vs vLLM selection score | 8 | 0.060 | 0.0518 |
| PEFT exact vs vLLM proposal exact | 8 | 0.060 | 0.0264 |
| PEFT exact vs vLLM default exact | 8 | 0.357 | 0.0156 |
| PEFT exact vs vLLM reordered exact | 8 | -0.535 | 0.0410 |

The default backend disagreement is not catastrophic row-by-row, but it is large
enough to break ranking:

```text
default vLLM-vs-PEFT common rows: 512
exact-label agreement: 96.4844%
text equality: 63.6719%
vLLM exact mean: 7.4219%
PEFT exact mean: 6.25%
```

The prompt-robust selector is therefore doing the wrong thing for this backend:
the reordered condition changes the candidate ordering substantially, and the
vLLM/PEFT default condition is not rank-stable enough to repair it.

## Verdict

This run supports a narrow systems-plus-quality claim, but not full dense
Gaussian perturbation parity:

```text
Supported:
  vLLM can screen this LoRA family much faster than all-candidate PEFT.
  vLLM proposal + PEFT confirmation gives 5.3x-6.7x full speedup at k<=8.
  PEFT-confirmed strict holdout matches/beats dense at k=4 and k=8 on this panel.

Not supported:
  vLLM robust selection is quality-preserving.
  q-only c2 is as powerful as dense Gaussian RandOpt at P64.
  the current shortlist policy can replace a dense PEFT screen.
  seed/spec-level dense Gaussian parity.
```

The next highest-leverage fix is not to scale P. It is to debug ranking
alignment: use PEFT-calibrated prompt conditions for shortlist scoring, and do
not include prompt variants whose vLLM rankings are anti-aligned with PEFT. A
larger shortlist is not justified by this run alone because dense-best recall
would still have needed `k > 32` under the current vLLM selection score.
