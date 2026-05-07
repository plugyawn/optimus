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

Dense Gaussian vs LoRA capacity audit:

```bash
python -m randopt_lora_lab.gaussian_parity \
  --out results/report/full_gaussian_parity \
  --rank 8
```

The first run is a correctness oracle. It must pass before any throughput or research claim.

## Evidence Rules

- Zero LoRA must match base outputs.
- Base outputs must match after candidate cycles.
- Log per-prompt rewards, not just aggregate means.
- Report cap-hit, malformed, and exact-answer rates separately.
- Compare lift vs base on the same prompt split.
- Built-in 32-example data is smoke-test only; use `--allow-repeat-data` only when repetition is intentional.
- Search and holdout splits must report zero overlap before a method ranking is treated as evidence.
- vLLM candidate selection is allowed only after adapter tensor parity and rank-correlation checks pass.
- Rank-`r` LoRA must not be described as full dense-Gaussian parity. Use `dense_gaussian`, `projected_gaussian_rank_r`, or `factor_gaussian_lora` as defined in `docs/full_gaussian_parity.md`.
