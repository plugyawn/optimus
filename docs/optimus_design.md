# Optimus Design

Optimus is a focused research library for zeroth-order post-training. The core
object is not a vLLM request or a LoRA adapter; it is a deterministic
perturbation that can be materialized by different backends.

## Scope

Optimus covers:

- perturbation identity, panels, and replay;
- dense and LoRA materialization;
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

The serving extra is for vLLM-backed LoRA adapter swapping. The eval extra
installs LightEval for standard and custom-task confirmation; combine eval and
serving extras for LightEval's vLLM backend.

## Package Layout

| package | responsibility |
| --- | --- |
| `optimus.core` | `PerturbationSpec`, deterministic panels, `ExperimentKey`, `RunRecord`, throughput records, and hooks. |
| `optimus.tasks` | Benchmark/task definitions, prompt construction, scoring, and split hygiene. |
| `optimus.commands` | Public CLI entrypoint modules. |
| `optimus.modeling` | Dense Gaussian patching, low-rank geometry, deterministic LoRA tensor generation, Qwen shape/config helpers, and PEFT-compatible adapter writing. |
| `optimus.search` | Backend-neutral zeroth-order study helpers, trusted Transformers search, selector scoring, and replay helpers. |
| `optimus.serving` | vLLM LoRA adapter-swapping search, halving, benchmark execution, prompt/token contracts, and output scoring. |
| `optimus.runs` | GPU workload specs, stable point identities, resumable execution logs, and plan serialization. |
| `optimus.evaluation` | Backend parity gates, LightEval command planning, run validation, systems reports, and release checks. |

## Perturbation Contract

`PerturbationSpec` is the boundary between search, modeling, serving, and
evaluation. It records the method (`dense` or `lora`), family, seed, scale,
sign, optional rank, and optional target modules. New manifests are JSONL
records with method-qualified keys; four-field candidate keys remain parseable
for older reports.

Backends must state the methods they support:

- vLLM search and benchmark paths accept only `lora` perturbations because vLLM
  hot-swaps LoRA adapters.
- Transformers search accepts `lora` through PEFT and `dense` through in-memory
  Gaussian patching.
- LightEval is a final-evaluation lane, not the full candidate-screening loop.

## Systems Contract

The library treats monitorability and throughput as part of the API:

- every GPU-suite point has a stable `ExperimentKey`;
- `optimus run-suite` records status, command, marker, return code, timestamps,
  and elapsed time for each point;
- summaries include candidate/sec, prompts/sec, tokens/sec, load time, eval
  elapsed time, tensor parallel size, batch/chunk settings, and method/backend;
- plots and CSVs keep selector quality, holdout-oracle quality, throughput, and
  staged-search regret separate.

This mirrors the operational discipline used for large training sweeps: do not
infer state from terminal history, do not mix budgets inside one claim, and make
failed or skipped points visible in machine-readable logs.

## Command Contract

Supported workflows use:

```bash
optimus perturbation-panel ...
optimus peft-search ...
optimus vllm-search ...
optimus vllm-halving ...
optimus vllm-bench ...
optimus backend-parity-gate ...
optimus run-plan ...
optimus run-suite ...
optimus validate-run ...
optimus systems-report ...
optimus lighteval ...
```

The repository root intentionally excludes old experiment namespaces, tracked
raw `results/`, and committed run bundles. Raw artifacts stay in ignored local
paths; docs describe the contract and summarize evidence without becoming a
paper-artifact dump.

## Evidence Gates

Quality claims require:

| evidence | gate |
| --- | --- |
| P1024/P4096 search | `summary.json`, `candidate_summary.jsonl`, per-prompt rows, heldout rows, and throughput fields. |
| Backend parity | Protocol match, base-row checks, ranking agreement, output-diff checks, and adapter tensor checks where applicable. |
| Dense/LoRA distinction | Rank-`r` LoRA is not claimed as dense parity unless a dense reference run actually passes. |
| LightEval confirmation | Standard-harness results and saved details for externally materialized final models. |
| Systems reporting | Backend/method-aware CSVs and PNGs for candidate/sec, token throughput, best-of-N, quality scaling, and staged search. |
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
