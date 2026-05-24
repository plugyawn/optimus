# Optimus Design

This document tracks the Optimus research-library structure.

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
python -m pip install -e ".[eval]"
python -m pip install -e ".[eval,serving]"
python -m pip install -e ".[dev]"
```

The serving extra is for vLLM-backed adapter swapping. The eval extra installs
LightEval for standard and custom-task confirmation; combine eval and serving
extras for LightEval's vLLM backend. Development and Prime bootstrap flows use
the dev extra instead of installing test tools ad hoc.

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
| `optimus.evaluation` | Backend parity gates, validity gates, LightEval planning, systems reports, plot generation. | Backend parity, run validation contracts, LightEval command planning, and systems-report APIs live here. |

## Command Contract

Supported workflows use:

```bash
optimus vllm-search ...
optimus peft-search ...
optimus backend-parity-gate ...
optimus run-plan ...
optimus run-suite ...
optimus validate-run ...
optimus systems-report ...
optimus lighteval ...
```

The repository root intentionally excludes old experiment namespaces and tracked
run-output directories. Repeatable evidence is committed as curated docs/report
artifacts, while raw run outputs stay in ignored workspace `results/` paths.

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

LightEval integration lives in `optimus.evaluation.lighteval`. Optimus still
uses vLLM adapter swapping for large candidate proposal panels; LightEval is the
standard-harness lane for final model/task confirmation, sample-level details,
and HF Hub dataset-backed evaluation.

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
| LightEval confirmation | Standard-harness final eval. | LightEval result files and saved details for selected base/LoRA candidates. |
| Throughput scaling | Candidate/sec, adapter throughput, and token/sec scaling. | `full_search_candidate_sec.png`, `adapter_throughput.png`, `token_throughput.png`, CSV inputs. |
| Best-of-N scaling | Running best score versus candidate count. | `best_of_n.csv`, `best_of_n.png`. |
| Quality scaling | Base, screen-selected heldout transfer, promoted holdout-oracle quality, and ensemble holdout quality versus population. | `quality_scaling.csv`, `quality_scaling.png`. |
| Run validation | Completeness gate for GPU suite outputs. | `optimus validate-run` JSON summary. |

## Documentation Coverage

| document | covers | gap |
| --- | --- | --- |
| `README.md` | Library purpose, install, workflows, evidence rules, and current P1024/P4096 evidence. | Current 4x evidence is linked; 8xA100 remains optional larger-systems evidence when provider inventory works. |
| `docs/api.md` | Supported package and CLI surface. | Should grow only when APIs become stable and tested. |
| `docs/index.md` | Public documentation map. | Keep top-level docs small and focused on supported workflows. |
| `docs/optimus_design.md` | Architecture and migration contract. | Must be updated as modules are physically moved. |
| `docs/gpu_suite.md` | P1024/P4096 GPU run contract, latest Prime evidence, and remaining validation gaps. | Needs staged-search and trusted-confirmation results for those specific claims. |
| `docs/reports/l40sx4_20260523_2237/report.md` | Current committed 4x L40S systems and quality evidence. | Staged search and trusted confirmation are not present in this bundle. |

## Completion Criteria

The Optimus refactor is not complete until:

1. Public commands use `optimus`.
2. Core library logic is organized under maintained package boundaries.
3. The repo root has no old experiment namespace and no tracked raw `results/`.
4. Unit tests cover the new public package surface.
5. P1024 and P4096 GPU runs produce the required quality, throughput, and scaling outputs.
6. Final design documentation matches the actual package tree and generated reports.
7. The release repo is upstreamed/presented as `optimus`.

Current status: items 1-7 are satisfied for the public Optimus library release.
The published package includes `optimus*` only, the final GitHub remote is
`plugyawn/optimus`, and the 4x L40S P1024/P4096 run provides the current
committed systems and quality evidence.

Remaining research extensions are staged-search savings and trusted
HF/PEFT/dense confirmation for promoted candidates. The intended 8xA100-class
run was attempted repeatedly; provider provisioning did not reach SSH, so the
accepted release evidence is the completed 4x L40S fallback.

`optimus release-check` is the machine-readable release gate for this list. It
checks package identity, public-doc old-name leakage, report selector/oracle
semantics, GPU artifact completeness, Prime cleanup ledger state, and the final
GitHub remote name.
