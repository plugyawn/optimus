# Optimus API

This page summarizes the supported Python and CLI surface. Optimus is a
zeroth-order optimization library: perturbation identity, materialization,
screening, validation, and systems reporting are separate contracts.

## Python Packages

| package | public use |
| --- | --- |
| `optimus.core` | `PerturbationSpec`, deterministic perturbation panels, experiment/run records, seed replay, and hook registry. |
| `optimus.tasks` | Countdown examples, prompt variants, exact-answer scoring, and split hygiene helpers. |
| `optimus.modeling` | Dense Gaussian patches, low-rank update geometry, and explicit export/materialization helpers. |
| `optimus.subspace` | Subspace basis state, candidate noise, and reference math. |
| `optimus.backends` | Backend names and compatibility boundaries. vLLM subspace search is the planned production backend and fails closed until Phase 5 lands. |
| `optimus.search` | Backend-neutral zeroth-order studies, prompt-condition scoring, and replay helpers. |
| `optimus.serving` | Prompt/token contracts and output scoring helpers used by backend integrations. |
| `optimus.runs` | GPU-suite run specs, stable experiment keys, plan serialization, and resumable execution records. |
| `optimus.evaluation` | Backend parity gates, run-output contracts, LightEval planning, validation, and systems reports. |

Use `PerturbationSpec`, `perturbation_panel`, and `read_perturbation_file` for
new integrations. Candidate parsing helpers that are not exported from public
packages are implementation details.

## CLI Commands

| command | purpose |
| --- | --- |
| `optimus perturbation-panel` | Write deterministic dense or LoRA perturbation panels. Subspace panels are produced by the subspace candidate schema and `run-plan`; this command fails closed for `--method subspace`. |
| `optimus search` | Run implemented perturbation search routes with `--backend vllm|transformers` and `--method dense|lora|subspace`; planned subspace routes fail closed until their roadmap phase lands. |
| `optimus bench` | Measure implemented backend throughput paths. `--backend vllm --method subspace` is reserved for the Phase 6 speed gate and fails closed until that backend lands. |
| `optimus make-countdown-data` | Generate deterministic Countdown evaluation data. |
| `optimus backend-parity-gate` | Gate vLLM selector trust against trusted outputs and adapter tensor checks. |
| `optimus lighteval` | Plan or run a LightEval job for standard or custom-task confirmation. |
| `optimus lighteval-sweep` | Plan or run LightEval across population-labelled materialized model artifacts. |
| `optimus systems-report` | Build LoRA/dense plot inputs and throughput/quality PNGs, and aggregate measured subspace `systems_report.json` files without inventing metrics. |
| `optimus run-plan` | Serialize the normalized P1024/P4096 GPU-suite plan. |
| `optimus run-suite` | Execute the normalized GPU-suite plan and write an execution log with stable point identities. |
| `optimus validate-run` | Validate that required GPU-suite outputs are present. |
| `optimus release-check` | Check package identity, public docs, report semantics, GPU artifact completeness, pod-ledger cleanup, and final GitHub remote identity. |

Historical experiment commands are not part of the supported source tree,
Optimus CLI, or published package.

## Perturbation Contract

`PerturbationSpec` records:

- `method`: `dense`, `lora`, or `subspace`;
- `family`: the sampling family, for example `dense_gaussian`,
  `isotropic`, `projected_gaussian_rank_r`, or
  `subspace_gaussian_rank_r`;
- `seed`, `sigma`, and `sign`: deterministic replay coordinates;
- optional `rank` and `targets` for rank-capped or adapter-exportable methods.

New keys are method-qualified, for example
`lora:isotropic:seed123:s0.0075:sign1` or
`subspace:subspace_gaussian_rank_r:seed123:s0.0075:sign1`.
Old method names are not part of the final public API.

Subspace families build a rank-capped activation basis from a fixed screen split
and sample per-candidate output noise. The implicit dense update is
`G @ activation_basis`. The planned vLLM backend applies this effect during
search without adapter swapping and fails closed until the roadmap acceptance
gates land. PEFT/vLLM adapter export is available only as a materialization step
for selected winners, using `subspace_state.pt`.

## Report Semantics

`optimus systems-report` treats selector quality and candidate-generation
quality as different quantities:

- `screen_selected_holdout_exact` is the heldout score of the candidate selected
  by the screen split.
- `promoted_holdout_oracle_exact` is the best heldout score among promoted
  candidates, and should be read as post-hoc candidate-generation evidence.
- quality claims about an operational selector should use the screen-selected
  heldout columns, not the holdout-oracle columns.

## Import Discipline

Metadata imports should not initialize Torch or vLLM:

```bash
python -c "import optimus.core, optimus.modeling, optimus.search, optimus.serving, optimus.evaluation"
```

Heavy backend imports are deferred until materialization, serving, or report
execution functions are called.
