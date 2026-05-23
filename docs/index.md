# Optimus Documentation

Optimus is a focused library for zeroth-order LoRA search on large language
models. The public documentation is intentionally small: it describes the
library surface, GPU validation contract, and operating discipline needed to
produce auditable research results.

## Core Documents

| document | purpose |
| --- | --- |
| `api.md` | Supported Python packages, CLI commands, and import discipline. |
| `optimus_design.md` | Package structure, API boundaries, migration contract, and completion criteria. |
| `gpu_suite.md` | Required P1024/P4096 workloads, acceptance gates, and output contracts. |
| `prime_gpu_runbook.md` | Prime GPU launch, sync, validation, and cleanup workflow. |
| `release_checklist.md` | Pre-release identity, evidence, systems, and upstreaming gates. |
| `reports/l40sx4_20260523_2237/` | Current committed 4x L40S P1024/P4096 report bundle and plot artifacts. |

## Archive

Historical validation notes live under `archive/experiments/`. They are useful
for audit trails, but they are not part of the public API or recommended user
workflow.
