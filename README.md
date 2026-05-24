# Optimus

Optimus is a research library for zeroth-order post-training of large language
models. The current focus is deterministic LoRA candidate search: generate a
large panel of candidate adapters, screen them quickly through vLLM adapter
swapping, and confirm winners with strict trusted-backend and heldout gates.

The supported public interface is the `optimus` package and CLI.

## What Optimus Is For

- High-throughput LoRA candidate screening with vLLM.
- Dense-reference and HF/PEFT confirmation of candidate quality.
- Population scaling studies for P1024/P4096 zeroth-order search.
- Systems plots: candidate/sec, prompts/sec, token throughput, adapter
  throughput, best-of-N scaling, quality scaling, and staged-search savings
  when a staged run is present.
- Auditable run outputs: candidate manifests, per-prompt rows, validation
  reports, parity gates, and plot inputs.

Optimus is intentionally narrow. It is not a general RLHF stack and it is not a
serving product; serving code exists to make candidate evaluation fast and
auditable.

## Install

```bash
python -m pip install -e .
```

GPU runs additionally need a CUDA build of PyTorch, vLLM with LoRA support, and
the model weights available locally or through Hugging Face authentication.

For vLLM serving work:

```bash
python -m pip install -e ".[serving]"
```

For LightEval-backed confirmation:

```bash
python -m pip install -e ".[eval]"
```

For LightEval confirmation through the vLLM backend:

```bash
python -m pip install -e ".[eval,serving]"
```

For local development:

```bash
python -m pip install -e ".[dev]"
```

## Package Structure

```text
optimus/
  core/        candidate identities, seed replay, shared types
  tasks/       benchmark data models, prompt builders, and scorers
  modeling/    model-specific adapter geometry and LoRA materialization
  runs/        reusable workload specifications for GPU validation suites
  search/      population construction and search utilities
  serving/     vLLM adapter-swapping execution
  evaluation/  reports, gates, and plot-oriented summaries
```

The documentation index is `docs/index.md`. The design contract is
`docs/optimus_design.md`, and the P1024/P4096 GPU suite runbook is
`docs/gpu_suite.md`.

## Core Commands

Generate Countdown data:

```bash
optimus make-countdown-data \
  --out data/countdown_generated_1200_seed20260507.json \
  --count 1200 \
  --seed 20260507
```

Run a vLLM LoRA search:

```bash
DATA=data/countdown_generated_1200_seed20260507.json

optimus vllm-search \
  --out results/p1024_vllm_search \
  --data "$DATA" \
  --model Qwen/Qwen2.5-3B-Instruct \
  --family isotropic \
  --population 1024 \
  --prompts 64 \
  --holdout-prompts 256 \
  --rank 8 \
  --targets q_proj,v_proj \
  --tensor-parallel-size 1 \
  --sigma-values 0.0075 \
  --max-new-tokens 32 \
  --stop-at-answer
```

Run a backend parity gate before trusting vLLM as selector of record:

```bash
optimus backend-parity-gate \
  --trusted results/backend_parity_gate/peft \
  --candidate results/backend_parity_gate/vllm \
  --out results/backend_parity_gate/gate
```

Build systems plots from existing runs:

```bash
optimus systems-report \
  --root results \
  --out results/report/optimus_systems
```

Write the planned P1024/P4096 workload:

```bash
optimus run-plan --out results/optimus_gpu_suite/plan.json
```

Execute the planned workload on a prepared GPU node:

```bash
optimus run-suite \
  --root results/optimus_gpu_suite \
  --systems-out results/report/optimus_systems \
  --populations 1024,4096
```

Validate a completed GPU run directory:

```bash
optimus validate-run \
  --root results/optimus_gpu_suite \
  --systems-out results/report/optimus_systems \
  --out results/optimus_gpu_suite/validation.json
```

Check release readiness:

```bash
optimus release-check \
  --gpu-root results/prime_runs/l40sx4_20260523_2237/results/optimus_gpu_suite_v092_noflash_tp4 \
  --systems-out results/prime_runs/l40sx4_20260523_2237/results/report/optimus_systems_v092_noflash_tp4 \
  --populations 1024,4096 \
  --bench-adapters 8 \
  --skip-halving \
  --strict
```

Plan a LightEval confirmation run:

```bash
optimus lighteval \
  --backend vllm \
  --model Qwen/Qwen2.5-3B-Instruct \
  --tensor-parallel-size 4 \
  --tasks ifeval \
  --out results/lighteval/ifeval_qwen25_3b \
  --plan-out results/lighteval/ifeval_qwen25_3b/plan.json
```

## Evidence Rules

Quality claims require:

- Unique train/screen/holdout examples with zero split overlap.
- Per-prompt rows, not only aggregate means.
- Exact-answer, malformed, cap-hit, and answer-closure rates reported separately.
- Prompt-robust selection when prompt variants are part of the claim.
- vLLM selection only after adapter tensor checks and ranking/output parity pass.
- Dense-reference, HF/PEFT, or LightEval-backed confirmation for promoted
  candidates.
- Clear separation between prompt-local lift, heldout lift vs base, and
  dense-Gaussian parity.

Rank-`r` LoRA should not be described as full dense-Gaussian parity unless the
run actually uses the dense reference family and passes the parity gate.

## Current Evidence Snapshot

The largest completed population-scale run is the Prime 4x L40S P1024/P4096
suite under:

```text
results/prime_runs/l40sx4_20260523_2237/results
```

The validated report is:

```text
results/prime_runs/l40sx4_20260523_2237/results/report/optimus_systems_v092_noflash_tp4/report.md
```

The same report artifacts are committed for public inspection under:

```text
docs/reports/l40sx4_20260523_2237/
```

The key quality result is deliberately reported in two columns:

- P1024 screen-selected heldout transfer: base `18/256`, selected `26/256`,
  lift `+8/256`.
- P4096 screen-selected heldout transfer: base `18/256`, selected `28/256`,
  lift `+10/256`.
- P1024 and P4096 promoted holdout-oracle candidates: `38/256`, lift
  `+20/256`.

That means the current run shows positive heldout transfer for the vLLM
screen-selected candidates and stronger candidate-generation capacity under the
post-hoc promoted holdout oracle. Optimus reports those separately so a
screen-selector claim cannot be hidden by a best-of-promoted number.

## GPU Suite Run Plan

The reference GPU workload is encoded in `scripts/run_optimus_gpu_suite.sh`:

1. P1024 full vLLM search with complete throughput and heldout outputs.
2. P4096 full vLLM search for best-of-N and scaling plots.
3. P1024 staged search to measure prompt-eval savings and selected-regret.
4. Trusted confirmation for promoted candidates.
5. `optimus validate-run` to verify the required run outputs.
6. A final report with best-of-N curves and MFU/scaling-style
   plots for zeroth-order optimization.

Prime GPU use must be logged in the project ledger before launch, and active
pods must be shut down or explicitly reported at the end of the run.

The 4x L40S run completed items 1, 2, 5, the adapter-throughput bench, and the
core report plots with `BENCH_ADAPTERS=8` and `RUN_HALVING=0`. The intended
8xA100 run was attempted, but provider provisioning never reached SSH; the 4x
fallback was used instead. Remaining publication-grade extensions are
staged-search evidence and trusted-backend confirmation for the promoted
candidates.
