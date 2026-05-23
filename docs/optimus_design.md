# Optimus Design

This document tracks the intended research-library structure while compatibility
modules are migrated into `optimus`.

## Scope

Optimus is a focused library for zeroth-order post-training research on LLMs. It
targets candidate-panel construction, LoRA materialization, high-throughput vLLM
evaluation, trusted confirmation, and research reporting. It is not a general
RLHF framework and it is not a model-serving product.

## Installation Surface

The base package installs the Python APIs and CPU-side reporting stack. Optional
extras keep operational dependencies explicit:

```bash
python -m pip install -e .
python -m pip install -e ".[serving]"
python -m pip install -e ".[dev]"
```

The serving extra is for vLLM-backed adapter swapping. Development and Prime
bootstrap flows use the dev extra instead of installing test tools ad hoc.

## Package Layout

| package | responsibility | current status |
| --- | --- | --- |
| `optimus.core` | Candidate identities, deterministic seed replay, shared utilities. | Candidate keys, candidate-panel generation, and module-name normalization live here without importing backend stacks. |
| `optimus.tasks` | Benchmark/task definitions, prompt construction, scoring, and split hygiene. | Countdown data model, prompt variants, exact scorer, and vote scorer live here. |
| `optimus.commands` | CLI entrypoint modules for search, validation, and reports. | Public command namespace added; user-facing reports now suggest `optimus ...` commands. |
| `optimus.modeling` | Model-specific adapter geometry, low-rank update geometry, deterministic LoRA noise families, and materialization helpers. | Qwen LoRA shape/config handling, low-rank projection helpers, deterministic candidate tensor generation, and PEFT-compatible adapter materialization live here; metadata imports stay lightweight. |
| `optimus.runs` | Reusable workload specs for GPU validation suites. | P1024/P4096 GPU suite plan generation added. |
| `optimus.search` | Population generation, trusted HF/PEFT search, prompt-condition scoring, staged search, reducer logic, shortlist construction. | Candidate-panel generation, trusted HF/PEFT search, and selector scoring live here; deeper reducer cleanup remains. |
| `optimus.serving` | vLLM adapter-swapping backend, HF/PEFT backend execution, prompt/token contracts, prompt input construction, output scoring, search/staged-search orchestration, and throughput-oriented execution. | Prompt/sampling contracts, HF/PEFT generation backends, vLLM prompt inputs, generation scoring, runtime helpers, and the vLLM search/halving/benchmark drivers live here. |
| `optimus.evaluation` | Backend parity gates, validity gates, dense confirmation, systems reports, plot generation. | Backend parity, run validation contracts, backend comparison, and systems-report APIs live here; deeper legacy audit migration remains. |
| legacy source namespace | Source-only migration layer for historical scripts and result manifests. | Excluded from the published package and Optimus CLI. |

## Compatibility Contract

Existing saved scripts and result manifests must keep working during migration.
New code should prefer:

```bash
optimus vllm-search ...
optimus peft-search ...
optimus backend-parity-gate ...
optimus run-plan ...
optimus run-suite ...
optimus validate-run ...
optimus systems-report ...
```

The migration rule is:

1. Expose public commands under `optimus.commands`.
2. Move reusable library logic into `optimus`.
3. Keep historical scripts source-only until saved replay manifests no longer
   depend on them.
4. Update scripts and docs to call `optimus` only after tests cover the new path.
5. Remove the source-only legacy package after archived workflows have either
   moved to `optimus` or been frozen outside the release tree.

## vLLM Search Core

The vLLM path is the primary systems contribution:

1. Load the base model once in vLLM.
2. Materialize deterministic PEFT-compatible LoRA adapters for each candidate.
3. Evaluate candidates through vLLM `LoRARequest` adapter swapping.
4. Save per-prompt rows, adapter manifests, environment metadata, and throughput.
5. Confirm winners through trusted HF/PEFT or dense-reference gates before making quality claims.

Migrated serving components now include Qwen adapter shape/config resolution in
`optimus.modeling.qwen`, low-rank update geometry in
`optimus.modeling.geometry`, update-family diagnostics in
`optimus.modeling.update_geometry`, deterministic LoRA noise families in
`optimus.modeling.noise`, dense Gaussian patching in `optimus.modeling.dense`,
PEFT-compatible adapter materialization in `optimus.modeling.lora`,
HF/PEFT Transformers execution in `optimus.serving.transformers`,
prompt/sampling contracts in
`optimus.serving.contracts`, prompt input construction in
`optimus.serving.prompting`, generation/output scoring helpers in
`optimus.serving.runtime`, the candidate-screen/holdout search driver in
`optimus.serving.search`, the staged-search driver in
`optimus.serving.halving`, and the adapter-throughput benchmark driver in
`optimus.serving.benchmark`. The serving commands expose
`--tensor-parallel-size`. The P1024/P4096 GPU suite plan defaults it to `8` for
8xA100 runs, and `optimus run-suite` executes the same run specs used to write
`plan.json`, so the shell launcher does not maintain a second copy of the
workload commands.

Trusted HF/PEFT confirmation lives in `optimus.search.peft`. It uses the same
candidate identity contract as the vLLM path and remains the reference backend
for candidate-quality confirmation and selector-parity gates.

LoRA aggregation/reducer utilities live in `optimus.search.aggregation`. The
core tensor construction is a reusable library API rather than command-local
implementation detail.

## Hooks

`optimus.core.hooks` provides a synchronous `HookRegistry` and structured
`OptimusEvent` objects. This is intentionally small: research scripts can attach
metrics logging, custom telemetry, candidate filters, or external schedulers
without baking those concerns into the serving path.

## GPU Suite Targets

The GPU suite runs should produce:

| run | purpose | required outputs |
| --- | --- | --- |
| P1024 full search | Quality and systems baseline. | `summary.json`, `candidate_summary.jsonl`, heldout rows, throughput, plots. |
| P4096 full search | Scaling and best-of-N curve. | Same as P1024 plus wall-clock/MFU-style summary. |
| P1024 staged search | Prompt-eval savings and top-candidate stability. | Halving report, regret vs full search, survivor recall. |
| Backend parity | Trust boundary for vLLM selector. | Tensor checks, ranking agreement, output diff, strict pass/fail report. |
| Throughput scaling | Candidate/sec, adapter throughput, and token/sec scaling. | `full_search_candidate_sec.png`, `adapter_throughput.png`, `token_throughput.png`, CSV inputs. |
| Best-of-N scaling | Running best score versus candidate count. | `best_of_n.csv`, `best_of_n.png`. |
| Quality scaling | Base, screen-selected heldout transfer, promoted holdout-oracle quality, and ensemble holdout quality versus population. | `quality_scaling.csv`, `quality_scaling.png`. |
| Run validation | Completeness gate for GPU suite outputs. | `optimus validate-run` JSON summary. |

## Documentation Coverage

| document | covers | gap |
| --- | --- | --- |
| `README.md` | Library purpose, install, workflows, evidence rules, and current P1024/P4096 evidence. | Needs 8xA100 rerun links after provider inventory is usable. |
| `docs/api.md` | Supported package and CLI surface. | Should grow only when APIs become stable and tested. |
| `docs/index.md` | Public documentation map. | Keep top-level docs small; promote archived notes only when they become maintained workflows. |
| `docs/optimus_design.md` | Architecture and migration contract. | Must be updated as modules are physically moved. |
| `docs/gpu_suite.md` | P1024/P4096 GPU run contract, latest Prime evidence, and remaining validation gaps. | Needs staged-search and trusted-confirmation results. |
| `docs/archive/experiments/` | Historical validation notes. | Provenance only; not part of the supported interface. |
| `results/prime_runs/l40sx2_20260523_2134/results/report/optimus_systems_v092_noflash/report.md` | Current 2x L40S systems and quality evidence. | Shows P4096 selector regret; needs 8xA100, staged search, and trusted confirmation. |

## Completion Criteria

The Optimus refactor is not complete until:

1. Public commands use `optimus`.
2. Core library logic is organized under maintained package boundaries.
3. Legacy names are compatibility-only and documented as such.
4. Unit tests cover the new public package surface.
5. P1024 and P4096 GPU runs produce the required quality, throughput, and scaling outputs.
6. Final design documentation matches the actual package tree and generated reports.
7. The release repo is upstreamed/presented as `optimus`, not as the old
   `randopt-lora-lab` experiment checkout.

Current status: items 1, 2, 4, and the basic P1024/P4096 throughput/report path
are partially satisfied. The published package now includes `optimus*` only,
while source-level legacy compatibility remains during migration. The goal is
still open because the latest large-population run exposed selector regret, the
8xA100/staged/trusted-confirmation evidence is not complete, and the final
GitHub upstreaming step has not been done.

`optimus release-check` is the machine-readable release gate for this list. It
checks package identity, public-doc legacy leakage, report selector/oracle
semantics, GPU artifact completeness, Prime cleanup ledger state, and the final
GitHub remote name.
