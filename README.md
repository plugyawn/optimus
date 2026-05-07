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
  --sigma-values 0.0005,0.001,0.002 \
  --promote 64 \
  --ensemble-ks 8,16,32,64 \
  --prompt-variants default,reordered \
  --score-mode robust_min \
  --min-selection-prompt-variants 2 \
  --stop-at-answer
```

Search scoring excludes prompt variants where the base model itself violates
the malformed/cap-hit protocol thresholds; those variants remain logged as
stress conditions. Method-quality claims still require multiple base-valid,
semantically equivalent prompt variants.

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
  --population 512 \
  --prompts 64 \
  --holdout-prompts 256 \
  --sigma-values 0.0005,0.001,0.002 \
  --promote 64 \
  --ensemble-ks 8,16,32,64 \
  --stop-at-answer
```

`top_holdout` is the old single-candidate diagnostic. Paper-style RandOpt
quality claims must read `ensemble_holdout`: candidates are selected by screen
score, the top-K candidates are evaluated on holdout, and Countdown votes by
valid numeric result rather than formula string.

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

Paper-style prompt/template mode for local reproduction checks:

```bash
OUT=results/paper_style_local \
POPULATION=128 \
PROMPTS=64 \
HOLDOUT_PROMPTS=256 \
SIGMA_VALUES=0.0005,0.001,0.002 \
PROMOTE=32 \
ENSEMBLE_KS=1,5,6,12,32 \
PROMPT_VARIANT=paper \
USE_CHAT_TEMPLATE=1 \
MAX_NEW_TOKENS=1024 \
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

HF/PEFT vs vLLM backend parity gate:

```bash
OUT_ROOT=results/backend_parity_gate \
FAMILY=factor_gaussian_lora \
POPULATION=64 \
PROMPTS=64 \
RANK=8 \
SIGMA=0.0075 \
scripts/run_backend_parity_gate.sh
```

This is the required gate before treating vLLM as the selector of record. It
runs the same candidate panel through the trusted HF/PEFT path and the vLLM
adapter path, keeps adapter files, checks sampled adapter tensors against the
canonical materializer, and then requires ranking correlation/top-k overlap.
Use `python -m randopt_lora_lab.backend_parity_gate` directly when comparing
two already-existing matched run directories.

Backend output diff for a failed gate:

```bash
python -m randopt_lora_lab.backend_output_diff \
  --trusted results/backend_parity_gate_p16/peft \
  --candidate results/backend_parity_gate_p16/vllm \
  --out results/backend_parity_gate_p16/output_diff
```

Next-token parity probe for a failed gate:

```bash
python -m randopt_lora_lab.backend_next_token_probe \
  --out results/backend_next_token_probe_p16 \
  --data "$DATA" \
  --prompts 8 \
  --seed 4242 \
  --rank 8 \
  --include-zero \
  --candidate factor_gaussian_lora:seed509771609:s0.0075:sign-1 \
  --candidate factor_gaussian_lora:seed1019282515:s0.0075:sign1
```

Short rollout parity probe for the same failed gate:

```bash
python -m randopt_lora_lab.backend_rollout_probe \
  --out results/backend_rollout_probe_p16 \
  --data "$DATA" \
  --prompts 8 \
  --seed 4242 \
  --rank 8 \
  --max-new-tokens 32 \
  --stop-at-answer \
  --include-zero \
  --candidate factor_gaussian_lora:seed509771609:s0.0075:sign-1 \
  --candidate factor_gaussian_lora:seed1019282515:s0.0075:sign1
```

Rollout parity ablation matrix:

```bash
ROOT=results/backend_rollout_ablation_p16 \
scripts/run_backend_rollout_ablation.sh
```

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
