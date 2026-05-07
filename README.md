# RandOpt LoRA Lab

Five-hour experiment lab for accelerating perturbation search on one A100/H100-class GPU.

The lab tests two axes:

- Systems: how fast can we evaluate many LoRA perturbation candidates without corrupting the base model?
- Geometry: can structured candidate distributions make useful perturbations appear more often?

Default task is Countdown with an answer-only expression prompt and exact expression reward.

## Run Targets

Generate or use a real non-repeated data file before any quality claim:

```bash
python -m randopt_lora_lab.make_countdown_data \
  --out data/countdown_generated_1200_seed20260507.json \
  --count 1200 \
  --seed 20260507
```

```bash
DATA=data/countdown_generated_1200_seed20260507.json

python -m randopt_lora_lab.experiments oracle --out results/oracle --data "$DATA" --prompts 64
python -m randopt_lora_lab.experiments search --out results/search_iso --data "$DATA" --family isotropic --population 64 --prompts 64 --holdout-prompts 256
python -m randopt_lora_lab.experiments search --out results/search_anzo --data "$DATA" --family anzo --population 64 --prompts 64 --holdout-prompts 256
python -m randopt_lora_lab.report --root results --out results/report
```

vLLM LoRA serving probe:

```bash
python -m randopt_lora_lab.vllm_lora_bench --out results/vllm_lora_bench --data "$DATA" --adapters 8 --prompts 64 --preload
```

Prompt-robust vLLM LoRA search:

```bash
python -m randopt_lora_lab.vllm_lora_search \
  --out results/vllm_robust_search \
  --data "$DATA" \
  --family sparse_low_rank_lora_d0p25 \
  --population 512 \
  --prompts 64 \
  --holdout-prompts 256 \
  --prompt-variants default,reordered \
  --score-mode robust_min \
  --stop-at-answer
```

Dense Gaussian vs LoRA capacity audit:

```bash
python -m randopt_lora_lab.gaussian_parity \
  --out results/report/full_gaussian_parity \
  --rank 8
```

Perturbation update-geometry audit:

```bash
python -m randopt_lora_lab.update_geometry \
  --out results/report/update_geometry \
  --rank 8 \
  --sigma 0.01
```

Slow trusted dense-Gaussian reference search:

```bash
python -m randopt_lora_lab.experiments search \
  --out results/dense_gaussian_ref \
  --data "$DATA" \
  --perturbation-backend dense \
  --family dense_gaussian \
  --population 64 \
  --prompts 64 \
  --holdout-prompts 256 \
  --stop-at-answer
```

Dense vs LoRA parity baseline:

```bash
OUT=results/gaussian_parity_baseline \
POPULATION=64 \
PROMPTS=64 \
HOLDOUT_PROMPTS=256 \
RANK=8 \
SIGMA=0.01 \
scripts/run_gaussian_parity_baseline.sh
```

Rank sweep:

```bash
BASE_OUT=results/gaussian_parity_rank_sweep \
RANKS=8,32 \
REUSE_DENSE=1 \
POPULATION=64 \
PROMPTS=64 \
HOLDOUT_PROMPTS=256 \
SIGMA=0.01 \
scripts/run_gaussian_parity_rank_sweep.sh
```

The baseline script runs `dense_gaussian`, `factor_gaussian_lora`, and
`projected_gaussian_rank_r`, then compares each LoRA-style arm against the same
dense panel. The projected arm is an SVD bridge baseline, not the fast path; it
is useful for separating "rank-r cannot carry the dense direction" from "our
factor-Gaussian sampling is the wrong low-rank distribution."
Rank sweeps default to reusing the first dense panel because dense Gaussian does
not depend on LoRA rank.

The first run is a correctness oracle. It must pass before any throughput or research claim.
The current completion checklist is in `docs/parity_completion_audit.md`.

Elite aggregate probe:

```bash
python -m randopt_lora_lab.aggregate_lora \
  --source-run results/gaussian_parity_rank_sweep_factor_only/rank8/lora \
  --out results/aggregate_rank8_top4 \
  --base-rank 8 \
  --top-k 4 \
  --weight-mode score \
  --data "$DATA" \
  --prompts 64 \
  --holdout-prompts 256 \
  --stop-at-answer
```

This tests whether a population can be used to form a single serveable adapter:
the weighted sum of `k` rank-`r` LoRA updates is represented exactly as one
rank-`k*r` adapter by concatenating the factors.

## Evidence Rules

- Zero LoRA must match base outputs.
- Base outputs must match after candidate cycles.
- Log per-prompt rewards, not just aggregate means.
- Report cap-hit, malformed, and exact-answer rates separately.
- Compare lift vs base on the same prompt split.
- Search winners must survive distinct protocol-valid prompt variants; repeated token caps do not count as separate prompt evidence.
- Built-in 32-example data is smoke-test only; use `--allow-repeat-data` only when repetition is intentional.
- Search and holdout splits must report zero overlap before a method ranking is treated as evidence.
- vLLM candidate selection is allowed only after adapter tensor parity and rank-correlation checks pass.
- Rank-`r` LoRA must not be described as full dense-Gaussian parity. Use `dense_gaussian`, `projected_gaussian_rank_r`, or `factor_gaussian_lora` as defined in `docs/full_gaussian_parity.md`.
