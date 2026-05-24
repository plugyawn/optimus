# Optimus

Optimus is a research library for zeroth-order post-training of large language
models. It gives perturbations stable identities, materializes them as dense
weight patches or LoRA adapters, runs high-throughput GPU screens, and keeps the
systems and validation records needed to decide whether a candidate actually
transfers.

The supported public interface is the `optimus` package and CLI.

## What Optimus Is For

- Backend-neutral perturbation panels with explicit `dense` and `lora` methods.
- High-throughput GPU screening through vLLM adapter swapping when the method is
  LoRA, and trusted Transformers execution for LoRA or dense perturbations.
- Population scaling studies for P1024/P4096 zeroth-order search.
- Systems plots: candidate/sec, prompts/sec, token throughput, backend/method
  throughput, best-of-N scaling, quality scaling, and staged-search savings
  when a staged run is present.
- Auditable run outputs: candidate manifests, per-prompt rows, validation
  reports, execution logs, parity gates, and plot inputs.

Optimus is intentionally narrow. It is not a general RLHF stack and it is not a
serving product; serving code exists to make candidate evaluation fast and
auditable.

## Install

```bash
python -m pip install -e .
```

GPU runs additionally need a CUDA build of PyTorch and the model weights
available locally or through Hugging Face authentication. vLLM is only required
for the adapter-swapping backend.

The default public model is `Qwen/Qwen3-4B`, a dense Qwen3 text model close to
the old 3B operating profile. Larger Qwen3.x models can be passed with
`--model`, but direct LoRA materialization is only validated for Qwen2, dense
Qwen3 text, and Qwen3-VL text configs.

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
  core/        perturbation identities, experiment records, seed replay
  tasks/       benchmark data models, prompt builders, and scorers
  modeling/    dense patches, low-rank geometry, and adapter materialization
  runs/        reusable workload specifications for GPU validation suites
  search/      zeroth-order studies, selection, and trusted HF execution
  serving/     vLLM adapter-swapping and throughput-oriented execution
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

Write a backend-neutral perturbation panel:

```bash
optimus perturbation-panel \
  --out results/panels/p1024_lora.jsonl \
  --method lora \
  --family isotropic \
  --population 1024 \
  --sigma 0.0075 \
  --rank 8 \
  --targets q_proj,v_proj \
  --seed 2468 \
  --antithetic
```

Run a trusted dense or LoRA search with Transformers:

```bash
optimus peft-search \
  --out results/p1024_dense_reference \
  --data data/countdown_generated_1200_seed20260507.json \
  --model Qwen/Qwen3-4B \
  --perturbation-backend dense \
  --family dense_gaussian \
  --population 1024 \
  --prompts 64 \
  --holdout-prompts 256 \
  --targets q_proj,v_proj \
  --sigma-values 0.0075 \
  --max-new-tokens 32 \
  --stop-at-answer
```

Run a high-throughput vLLM LoRA search:

```bash
DATA=data/countdown_generated_1200_seed20260507.json

optimus vllm-search \
  --out results/p1024_vllm_search \
  --data "$DATA" \
  --model Qwen/Qwen3-4B \
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
  --gpu-root results/optimus_gpu_suite \
  --systems-out results/report/optimus_systems \
  --populations 1024,4096 \
  --bench-adapters 8 \
  --strict
```

Plan a LightEval confirmation run:

```bash
optimus lighteval \
  --backend vllm \
  --model Qwen/Qwen3-4B \
  --data-parallel-size 8 \
  --max-model-length 4096 \
  --tasks ifeval \
  --out results/lighteval/ifeval_qwen3_4b \
  --plan-out results/lighteval/ifeval_qwen3_4b/plan.json
```

Plan a matched LightEval sweep for materialized population artifacts:

```bash
optimus lighteval-sweep \
  --backend vllm \
  --tasks ifeval \
  --model-template results/materialized/p{population} \
  --populations 128,256,512,1024,4096 \
  --data-parallel-size 8 \
  --max-model-length 4096 \
  --out-root results/lighteval/population_sweep \
  --plan-out results/lighteval/population_sweep/plan.json
```

## Evidence Rules

Quality claims require:

- Unique train/screen/holdout examples with zero split overlap.
- Per-prompt rows, not only aggregate means.
- Exact-answer, malformed, cap-hit, and answer-closure rates reported separately.
- Prompt-robust selection when prompt variants are part of the claim.
- vLLM selection only after adapter tensor checks and ranking/output parity pass.
- Dense-reference, HF/PEFT, or LightEval-backed confirmation for the selected
  materialized model state.
- Clear separation between prompt-local lift, heldout lift vs base, and
  dense-Gaussian parity.

Rank-`r` LoRA should not be described as full dense-Gaussian parity unless the
run actually uses the dense reference family and passes the parity gate.

## Evidence Snapshot

Optimus keeps concrete run artifacts under ignored local `results/` paths.
Public claims should be regenerated from those artifacts with
`optimus systems-report` and `optimus release-check`; the repository does not
ship stale run bundles as source files.

Reports must separate selector evidence from post-hoc candidate-generation
evidence. The screen-selected holdout column is the selector claim. The promoted
holdout-oracle column is only candidate-generation evidence and is not a
substitute for trusted confirmation.

## GPU Suite Run Plan

The reference GPU workload is encoded in `scripts/run_optimus_gpu_suite.sh`:

1. P1024 full perturbation search with complete throughput and heldout outputs.
2. P4096 full perturbation search for best-of-N and scaling plots.
3. P1024 staged search to measure prompt-eval savings and selected-regret.
4. Trusted confirmation for selected candidates or final materialized models.
5. `optimus validate-run` to verify the required run outputs.
6. A final report with best-of-N curves and MFU/scaling-style
   plots for zeroth-order optimization.

Prime GPU use must be logged in the project ledger before launch, and active
pods must be shut down or explicitly reported at the end of the run.

Publication-grade claims additionally require staged-search evidence when
staging is claimed and a passing trusted-backend or LightEval confirmation for
the selected model state.
