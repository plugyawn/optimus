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

The quality gate failed. vLLM did not shortlist the dense best candidate:

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

## Verdict

This run supports only the systems half of the claim:

```text
Supported:
  vLLM can screen this LoRA family much faster than all-candidate PEFT.
  vLLM proposal + PEFT confirmation gives 5.3x-6.7x full speedup at k<=8.

Not supported:
  vLLM robust selection is quality-preserving.
  q-only c2 is as powerful as dense Gaussian RandOpt at P64.
  the current shortlist policy can replace a dense PEFT screen.
```

The next highest-leverage fix is not to scale P. It is to debug ranking
alignment: compare PEFT and vLLM outputs for the same q-only candidates across
the same prompt variants, then select using a PEFT-calibrated proposal score or
increase shortlist size only if the dense-best recall curve justifies it.
