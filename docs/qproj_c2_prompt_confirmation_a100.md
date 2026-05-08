# QProj C2 Prompt Confirmation A100

## Run

```text
date: 2026-05-08
pod: qproj-c2-prompt-confirm, Prime pod 62c6857728c64acfb3b22eac3882ef53
gpu: 1x NVIDIA A100-SXM4-80GB
source run: results/activation_spectral_qv_split_p32_a100/q_proj_activation_spectral_lora_c2
out: results/qproj_c2_prompt_confirmation
model: Qwen/Qwen2.5-3B-Instruct
family: activation_spectral_lora_c2
targets: q_proj
rank: 32
top-k: 4 strict numeric vote
holdout prompts: 128
caps: 64, 128, 256
prompt variants: default, reordered, xml
```

The goal was to check whether the q-only c2 signal from the Q/V split audit was
only a default-prompt artifact. The gate required at least two prompt variants
to be protocol-valid, quality-valid, and above a 1.5625 percentage-point lift at
all tested caps.

## Result

The strict top-4 ensemble gate passed on the two base-valid prompt variants:

```text
gate pass: true
valid prompt variants: 2
passing prompt variants: 2
valid prompt conditions: 6
invalid prompt conditions: 3
min lift observed: 2.34375 pp
mean lift observed: 3.90625 pp
```

| prompt | cap | strict top-4 | base | lift | max malformed | max cap-hit | valid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| default | 64 | 9.375% | 7.031% | +2.344 pp | 3.906% | 0.000% | true |
| default | 128 | 9.375% | 7.031% | +2.344 pp | 3.906% | 0.000% | true |
| default | 256 | 9.375% | 7.031% | +2.344 pp | 3.906% | 0.000% | true |
| reordered | 64 | 10.938% | 5.469% | +5.469 pp | 4.688% | 0.781% | true |
| reordered | 128 | 10.938% | 5.469% | +5.469 pp | 4.688% | 0.781% | true |
| reordered | 256 | 10.938% | 5.469% | +5.469 pp | 4.688% | 0.781% | true |
| xml | 64 | 4.688% | 3.906% | +0.781 pp | 35.938% | 2.344% | false |
| xml | 128 | 4.688% | 3.906% | +0.781 pp | 35.938% | 2.344% | false |
| xml | 256 | 4.688% | 3.906% | +0.781 pp | 35.938% | 2.344% | false |

The XML variant is not negative evidence for the perturbation family because the
base run itself is outside the malformed threshold: base malformed was 26.5625%
and base cap-hit was 0.78125%. It should be treated as a protocol-invalid prompt
variant, not as a valid prompt-robust comparison.

## Verdict

This is the first clean prompt/cap positive for the local LoRA family:

```text
q-only activation_spectral_lora_c2:
  prompt/cap strict ensemble gate: pass
  supported prompt variants: default, reordered
  unsupported prompt variant: xml, due base protocol invalidity
```

It still does not prove that the family is as powerful as dense Gaussian
RandOpt. The next required test is a systems-quality confirmation: use vLLM as a
fast proposal engine for this q-only c2 family, PEFT-confirm the shortlist, and
compare recovered quality and regret against a matched dense Gaussian reference.
