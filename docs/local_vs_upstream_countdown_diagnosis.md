# Local vs Upstream Countdown Diagnosis

Date: 2026-05-07

## Why This Exists

The local dense/LoRA panels can look suspicious because one panel reports:

```text
base holdout exact: 8.203125%
best dense holdout exact: 8.203125%
```

That is real for `results/perf_quality_p64_rank8/dense`, but it is not evidence
about the official RandOpt Countdown setup. It is a local stress panel:

```text
model: Qwen/Qwen2.5-3B-Instruct
targets: q_proj,v_proj
population: 64
screen prompts: 64
holdout prompts: 256
sigma: 0.01
max_new_tokens: 32
reward: exact-answer only
```

On that run, the top dense screen candidate was:

```text
dense_gaussian:seed410114451:s0.01:sign1
screen exact: 9.375%
holdout exact: 8.203125%
base holdout exact: 8.203125%
```

So the right interpretation is screen overfit / sparse exact-reward noise on a
small local panel, not a failed reproduction of the RandOpt paper.

## Current Reproduction Audit

The refreshed official-style audit for
`results/paper_style_p128_qwen3b/dense` fails:

```text
pass: false
failed:
  model
  official_countdown_data
  full_parameter_targets
  dense_noise_mode
  candidate_score_metric
  ensemble_vote_metric
  train_or_screen_samples
  population
  max_new_tokens
  prompt_variant
  use_chat_template
  ensemble_ks
```

The local `paper_style_p128_qwen3b` run is cleaner than the P64 exact-only
stress panel, but it is still not an official reproduction:

```text
model: Qwen/Qwen2.5-3B-Instruct, not allenai/Olmo-3-7B-Instruct
data: generated local Countdown shard, not upstream official Countdown JSON
targets: q_proj,v_proj, not all non-visual parameters
population: 128, not 5000
screen prompts: 64, not 200
max_new_tokens: 128, not 1024
prompt/template: not the paper prompt with chat template
score/vote: not upstream reward plus numeric-vote semantics
```

## Upstream Smoke Sanity Check

The actual upstream path was smoke-tested at small population in
`docs/upstream_official_p32_smoke.md`:

```text
upstream repo: sunrainyg/RandOpt
model: allenai/Olmo-3-7B-Instruct
data: upstream official Countdown JSON
train samples: 200
test samples: 128
population: 32
sigmas: 0.0005,0.001,0.002
max tokens: 1024
```

That smoke found positive train-reward candidates:

```text
base train reward: 72.715%
base test reward: 67.586%
best candidate train reward: 79.420%
K=3 numeric-vote smoke eval: 71.094%
K=1 smoke eval: 64.844%
```

This validates that the upstream machinery can produce the expected kind of
positive RandOpt signal at small P. It does not prove paper-scale reproduction
because the official population is `5000`.

## Operating Rule

Use these labels consistently:

```text
official reproduction
  Only if reproduction_audit passes.

upstream smoke
  Official upstream semantics but reduced population or eval split.

local parity panel
  Qwen/local data/local reward/local target set; useful for dense-vs-LoRA
  engineering comparisons, not for paper reproduction claims.
```

The local parity panels remain useful for the project goal because they compare
dense Gaussian and serveable LoRA-style perturbation families under controlled
local conditions. They should not be used to judge whether the RandOpt paper was
reproduced.

## Next Quality Test

The highest-leverage local quality test is still not another factor-LoRA rerun.
It is the projected bridge:

```text
dense_gaussian
projected_gaussian_rank_r
factor_gaussian_lora
```

`projected_gaussian_rank_r` samples the same dense Gaussian direction and uses
the best rank-r SVD projection as a LoRA adapter. This separates two questions:

```text
1. Is rank-r too small to preserve the useful dense direction?
2. Or is factor_gaussian_lora simply the wrong low-rank distribution?
```

Scaling should remain blocked unless the projected/factor arms are evaluated
against the same dense candidate panel, with disjoint screen/holdout prompts,
validity audits, and cap/malformed/token-count reporting.
