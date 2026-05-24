# Optimus API

This page summarizes the supported Python and CLI surface. The API is intentionally
small: Optimus is for zeroth-order LoRA candidate search, vLLM-backed screening,
trusted confirmation, and run reporting.

## Python Packages

| package | public use |
| --- | --- |
| `optimus.core` | Candidate identity, deterministic panel generation, and hook registry. |
| `optimus.tasks` | Countdown examples, prompt variants, exact-answer scoring, and split hygiene helpers. |
| `optimus.modeling` | Qwen LoRA shape resolution, low-rank update geometry, deterministic LoRA noise families, and PEFT-compatible adapter materialization. |
| `optimus.search` | Search-candidate construction, trusted HF/PEFT search, prompt-condition scoring, and replay helpers. |
| `optimus.serving` | vLLM adapter-swapping search, HF/PEFT backend execution, staged search, and benchmark orchestration, prompt/token contracts, prompt input builders, and output scoring helpers. |
| `optimus.runs` | P1024/P4096 run specs, plan serialization, and suite execution helpers. |
| `optimus.evaluation` | Backend parity gates, run-output contracts, validation, and report generation entry points. |

## CLI Commands

| command | purpose |
| --- | --- |
| `optimus make-countdown-data` | Generate deterministic Countdown evaluation data. |
| `optimus vllm-search` | Screen LoRA candidates with vLLM adapter swapping. |
| `optimus vllm-halving` | Run staged candidate screening. |
| `optimus vllm-bench` | Measure adapter materialization and adapter-swapping throughput. |
| `optimus peft-search` | Run trusted HF/PEFT candidate evaluation. |
| `optimus backend-parity-gate` | Gate vLLM selector trust against trusted outputs. |
| `optimus lighteval` | Plan or run a LightEval job for standard or custom-task confirmation. |
| `optimus systems-report` | Build plot inputs and throughput/quality PNGs from result directories. |
| `optimus run-plan` | Serialize the normalized P1024/P4096 GPU-suite plan. |
| `optimus run-suite` | Execute the normalized GPU-suite plan and write an execution log. |
| `optimus validate-run` | Validate that required GPU-suite outputs are present. |
| `optimus release-check` | Check package identity, public docs, report semantics, GPU artifact completeness, pod-ledger cleanup, and final GitHub remote identity. |

Historical experiment commands are not part of the supported source tree,
Optimus CLI, or published package.

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
python -c "import optimus.core, optimus.modeling, optimus.serving, optimus.evaluation"
```

Heavy backend imports are deferred until materialization, serving, or report
execution functions are called.
