# Elite Aggregate LoRA Probe

Source run:

```text
results/gaussian_parity_rank_sweep_factor_only/rank8/lora
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
base rank: 8
aggregate rank: 32
```

Result:

```text
base screen exact:        6.250%
base holdout exact:       8.984%
aggregate screen exact:   0.000%
aggregate holdout exact:  1.172%
holdout lift:            -7.812 pp
holdout cap-hit:          1.562%
holdout malformed:        1.953%
```

Verdict:

```text
fail
```

Rank-8 score-weighted top-4 aggregation is worse than base and does not provide
a useful sample-size aggregation path.
