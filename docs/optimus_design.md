# Optimus Design

Optimus is a focused research library for zeroth-order post-training. The core
object is not a vLLM request or a LoRA adapter; it is a deterministic
perturbation that can be materialized by different backends.

## Scope

Optimus covers:

- perturbation identity, panels, and replay;
- dense, LoRA, and subspace materialization/replay;
- high-throughput GPU screening;
- trusted confirmation through Transformers, dense-reference checks, backend
  parity, and LightEval;
- run records, throughput metrics, and evidence reports.

It is not a general RLHF framework and it is not a serving product. Serving code
exists to make candidate evaluation fast, reproducible, and inspectable.

## Installation Surface

The base package installs Python APIs and CPU-side reporting. Optional extras
keep operational dependencies explicit:

```bash
python -m pip install -e .
python -m pip install -e ".[serving]"
python -m pip install -e ".[eval]"
python -m pip install -e ".[eval,serving]"
python -m pip install -e ".[dev]"
```

The serving extra is for vLLM-backed LoRA adapter serving and for the planned
production subspace backend once Phase 5 lands. The eval extra installs
LightEval for standard and custom-task confirmation; combine eval and serving
extras for LightEval's vLLM backend.

## Package Layout

| package | responsibility |
| --- | --- |
| `optimus.core` | `PerturbationSpec`, deterministic panels, `ExperimentKey`, `RunRecord`, throughput records, and hooks. |
| `optimus.tasks` | Benchmark/task definitions, prompt construction, scoring, and split hygiene. |
| `optimus.commands` | Public CLI entrypoint modules. |
| `optimus.modeling` | Dense Gaussian patching, low-rank geometry, Qwen shape/config helpers, and explicit export/materialization helpers. |
| `optimus.subspace` | Subspace basis state, candidate noise, and reference math. |
| `optimus.backends` | Backend names and compatibility boundaries; vLLM subspace search is planned and fails closed until Phase 5. |
| `optimus.search` | Backend-neutral zeroth-order study helpers, selector scoring, and replay helpers. |
| `optimus.serving` | Prompt/token contracts and output scoring helpers used by backend integrations. |
| `optimus.runs` | GPU workload specs, stable point identities, resumable execution logs, and plan serialization. |
| `optimus.evaluation` | Backend parity gates, LightEval command/sweep planning, run validation, systems reports, and release checks. |

## Perturbation Contract

`PerturbationSpec` is the boundary between search, modeling, serving, and
evaluation. It records the method (`dense`, `lora`, or `subspace`), family,
seed, scale, sign, optional rank, and optional target modules. New manifests
are JSONL records with method-qualified keys.

Backends must state the methods they support:

- `--backend vllm --method subspace` is the planned production subspace search
  path and currently fails closed with a Phase 5 pointer.
- `--backend vllm --method lora` is adapter-search/replay infrastructure.
- `--backend transformers` is the trusted reference path for dense, LoRA, and
  subspace checks.
- Subspace PEFT/vLLM adapter export is an explicit materialization step for
  selected winners, not the search hot path.
- LightEval is a final-evaluation lane, not the full candidate-screening loop.

The transformer-linear subspace lazy-kernel implementation is governed by
`full_model_lazy_kernel_design.md`. That design keeps vLLM as the production
execution substrate while Optimus owns only the `G_t,c Q_s x` perturbation
operator, candidate state, artifact contract, and validation gates.

## Systems Contract

The library treats monitorability and throughput as part of the API:

- every GPU-suite point has a stable `ExperimentKey`;
- `optimus run-suite` records status, command, marker, return code, timestamps,
  and elapsed time for each point;
- summaries include candidate/sec, prompts/sec, tokens/sec, load time, eval
  elapsed time, tensor parallel size, batch/chunk settings, and method/backend;
- plots and CSVs keep selector quality, holdout-oracle quality, and throughput
  separate. Historical staged-search artifacts are parsed separately when
  present, but staged search is not a supported public route yet.

This mirrors the operational discipline used for large training sweeps: do not
infer state from terminal history, do not mix budgets inside one claim, and make
failed or skipped points visible in machine-readable logs.

## Command Contract

Supported workflows use:

```bash
optimus perturbation-panel ...
optimus search --backend vllm --method subspace ...
optimus search --backend vllm --method lora ...
optimus search --backend transformers --method dense ...
optimus bench --backend vllm --method subspace ...
optimus backend-parity-gate ...
optimus run-plan ...
optimus run-suite ...
optimus validate-run ...
optimus systems-report ...
optimus lighteval ...
optimus lighteval-sweep ...
```

The repository root intentionally excludes old experiment namespaces, tracked
raw `results/`, and committed run bundles. Raw artifacts stay in ignored local
paths; docs describe the contract and summarize evidence without becoming a
paper-artifact dump.

## Evidence Gates

Quality claims require:

| evidence | gate |
| --- | --- |
| Subspace P1024/P4096 search | `subspace_state.pt`, `subspace_state_summary.json`, `candidates.jsonl`, `candidate_scores.jsonl`, `top_k_ensemble.json`, `validation_report.json`, `systems_report.json`, `summary.json`, sample-level scorer details, and throughput fields. |
| Legacy LoRA baseline search | Explicitly LoRA-labelled `summary.json`, `candidate_summary.jsonl`, per-prompt rows, heldout rows, adapter metadata, and throughput fields. |
| Backend parity | Protocol match, base-row checks, ranking agreement, output-diff checks, and adapter tensor checks where applicable. |
| Dense/LoRA distinction | Rank-`r` LoRA is not claimed as dense parity unless a dense reference run actually passes. |
| LightEval confirmation | Standard-harness results and saved details for externally materialized final models; lazy top-K ensembles use Optimus-native sample-level evaluation until a direct LightEval runtime adapter exists. |
| Systems reporting | Backend/method-aware CSVs and PNGs for candidate/sec, token throughput, best-of-N, and quality scaling. Staged artifacts are legacy report inputs only until a final staged route exists. |
| GPU operations | Execution log, run validation, and pod cleanup ledger. |

## Completion Criteria

The library structure is acceptable when:

1. public commands use `optimus`;
2. core logic is organized under maintained package boundaries;
3. the repo root has no old experiment namespace, tracked raw results, or
   committed report bundles;
4. unit tests cover perturbation identity, method routing, run records, import
   discipline, and release gates;
5. GPU-suite outputs preserve stable point identity and systems metrics;
6. docs match the actual package tree and do not present old scripts as the
   library interface;
7. the GitHub remote is `optimus`.

`optimus release-check` is the machine-readable release gate for package
identity, public-doc old-name leakage, report semantics, GPU artifact
completeness, Prime cleanup state, and GitHub remote identity.
