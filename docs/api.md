# Optimus API

This page summarizes the supported Python and CLI surface. Optimus is a
zeroth-order optimization library: perturbation identity, materialization,
screening, validation, and systems reporting are separate contracts.

## Python Packages

| package | public use |
| --- | --- |
| `optimus.core` | `PerturbationSpec`, deterministic perturbation panels, experiment/run records, seed replay, and hook registry. |
| `optimus.tasks` | Countdown examples, prompt variants, exact-answer scoring, and split hygiene helpers. |
| `optimus.modeling` | Dense Gaussian patches, low-rank update geometry, deterministic LoRA noise families, and PEFT-compatible adapter materialization. |
| `optimus.search` | Backend-neutral zeroth-order studies, trusted Transformers search, prompt-condition scoring, and replay helpers. |
| `optimus.serving` | vLLM LoRA adapter-swapping search, staged search, benchmark orchestration, prompt/token contracts, and output scoring helpers. |
| `optimus.runs` | GPU-suite run specs, stable experiment keys, plan serialization, and resumable execution records. |
| `optimus.evaluation` | Backend parity gates, run-output contracts, LightEval planning, validation, and systems reports. |

Use `PerturbationSpec`, `perturbation_panel`, and `read_perturbation_file` for
new integrations. Legacy candidate helpers live only in their internal module
for old manifest parsing and are not part of the public package exports.

## CLI Commands

| command | purpose |
| --- | --- |
| `optimus perturbation-panel` | Write a deterministic dense or LoRA perturbation panel. |
| `optimus peft-search` | Run trusted Transformers evaluation with `--perturbation-backend lora` or `dense`. |
| `optimus vllm-search` | Screen LoRA perturbations with vLLM adapter swapping. |
| `optimus vllm-halving` | Run staged LoRA screening with vLLM. |
| `optimus vllm-bench` | Measure adapter materialization and adapter-swapping throughput. |
| `optimus make-countdown-data` | Generate deterministic Countdown evaluation data. |
| `optimus backend-parity-gate` | Gate vLLM selector trust against trusted outputs and adapter tensor checks. |
| `optimus lighteval` | Plan or run a LightEval job for standard or custom-task confirmation. |
| `optimus systems-report` | Build backend/method-aware plot inputs and throughput/quality PNGs from result directories. |
| `optimus run-plan` | Serialize the normalized P1024/P4096 GPU-suite plan. |
| `optimus run-suite` | Execute the normalized GPU-suite plan and write an execution log with stable point identities. |
| `optimus validate-run` | Validate that required GPU-suite outputs are present. |
| `optimus release-check` | Check package identity, public docs, report semantics, GPU artifact completeness, pod-ledger cleanup, and final GitHub remote identity. |

Historical experiment commands are not part of the supported source tree,
Optimus CLI, or published package.

## Perturbation Contract

`PerturbationSpec` records:

- `method`: `dense` or `lora`;
- `family`: the sampling family, for example `dense_gaussian`,
  `isotropic`, or `projected_gaussian_rank_r`;
- `seed`, `sigma`, and `sign`: deterministic replay coordinates;
- optional `rank` and `targets` for adapter-style methods.

New keys are method-qualified, for example
`lora:isotropic:seed123:s0.0075:sign1`. Legacy four-part keys are still parsed
for old manifests.

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
