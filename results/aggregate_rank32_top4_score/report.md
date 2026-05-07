# Elite Aggregate LoRA Probe

Source runs:

```text
results/gaussian_parity_rank_sweep_factor_only/rank8/lora
results/gaussian_parity_rank_sweep_factor_only/rank32/lora
```

Setup:

```text
model: Qwen/Qwen2.5-3B-Instruct
task: generated Countdown
screen prompts: 64
holdout prompts: 256
max_new_tokens: 32
stop_at_answer: true
aggregate mode: concatenate LoRA factors, weight B factors
elite count: top 4 by screen score
weight mode: score
```

## Results

| base rank | aggregate rank | base holdout exact | aggregate screen exact | aggregate holdout exact | holdout lift | holdout cap-hit | holdout malformed | verdict |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 8 | 32 | 8.984% | 0.000% | 1.172% | -7.812 pp | 1.562% | 1.953% | fail |
| 32 | 128 | 8.984% | 9.375% | 14.062% | +5.078 pp | 25.781% | 1.953% | promising but invalidated by cap-hit risk |

The rank-8 aggregate is a clean negative. It collapses below base on both
screen and holdout.

The rank-32 aggregate is the first positive sample-size aggregation signal in
this lab: combining four high-screen rank-32 LoRA perturbations beats base
holdout and beats the best individual rank-32 factor-LoRA holdout observed in
the source run.

However, the rank-32 aggregate also raises holdout cap-hit to 25.781%. That is
too high for a quality claim. The next gate is cap-stability with higher token
caps and an answer-only prompt before any optimizer or systems claim is made.

## Immediate Next Gate

Run only the selected aggregate and its four elite constituents:

```text
max_new_tokens: 32, 64, 128, 256
prompt variants: current, answer-only
metrics: exact, extracted answer, valid equation, output tokens, finish reason
```

Pass condition:

```text
The rank-32 aggregate remains above base and above its elite constituents while
cap-hit goes near zero and malformed rate does not regress.
```

If the lift disappears when the cap is raised or the prompt is tightened, this
was mostly a truncation/format artifact. If it survives, aggregation becomes the
highest-priority "large sample unlock" path.
