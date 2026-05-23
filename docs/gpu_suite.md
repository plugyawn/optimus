# Optimus GPU Suite

This runbook defines the P1024/P4096 GPU workloads used to validate Optimus
search quality, serving throughput, and staged-search behavior.

## Required Runs

| run | default output | purpose |
| --- | --- | --- |
| P1024 full search | `results/optimus_gpu_suite/search_p1024_chunk8` | Matched quality and systems baseline. |
| P4096 full search | `results/optimus_gpu_suite/search_p4096_chunk8` | Best-of-N and scaling evidence. |
| P1024 halving | `results/optimus_gpu_suite/halving_p1024_stage8_surv64` | Staged-search savings and regret. |
| Adapter throughput benches | `results/optimus_gpu_suite/bench_a*_p64` | Candidate/sec, prompts/sec, tokens/sec, adapter-scaling data. |
| Systems report | `results/report/optimus_systems` | Plot inputs and PNGs for candidate/sec, adapter throughput, token throughput, best-of-N, quality scaling, and staging tradeoffs. |
| Execution log | `results/optimus_gpu_suite/execution.json` | Ordered command/status record from `optimus run-suite`. |

## Launcher

```bash
scripts/run_optimus_gpu_suite.sh
```

The launcher writes the normalized workload to:

```text
results/optimus_gpu_suite/plan.json
```

You can generate the plan without launching GPU work:

```bash
optimus run-plan --out results/optimus_gpu_suite/plan.json
```

The launcher delegates execution to the same Optimus run specs:

```bash
optimus run-suite \
  --root results/optimus_gpu_suite \
  --systems-out results/report/optimus_systems \
  --populations 1024,4096 \
  --execution-log results/optimus_gpu_suite/execution.json
```

Important overrides:

```bash
OUT_ROOT=results/optimus_gpu_suite \
POPULATIONS="1024 4096" \
PROMPTS=64 \
HOLDOUT_PROMPTS=256 \
PROMOTE=64 \
CHUNK_ADAPTERS=8 \
MAX_LORAS=8 \
MAX_CPU_LORAS=8192 \
TENSOR_PARALLEL_SIZE=8 \
scripts/run_optimus_gpu_suite.sh
```

## Acceptance Gates

The generated evidence is not enough by itself. A final claim requires:

1. `summary.json` and per-prompt rows for each full search.
2. Candidate/sec, prompts/sec, token/sec, load time, and eval elapsed time.
3. Heldout evaluation for promoted candidates.
4. Backend parity or trusted confirmation before using vLLM ranking as the
   selector of record.
5. P1024/P4096 best-of-N curves from saved candidate summaries.
6. Systems plots from `optimus systems-report`, including `adapter_throughput.png`, `token_throughput.png`, `best_of_n.png`, and `quality_scaling.png`.
7. `optimus validate-run` passes for the run root and systems report.
8. Active GPU pods stopped or explicitly reported after the run.

## Current Validation State

The software path has passed local validation, a remote L40S bootstrap smoke,
and a completed Prime 4x L40S P1024/P4096 run.

Latest completed run:

```text
results/prime_runs/l40sx4_20260523_2237/results
```

Run shape:

- hardware: Prime Crusoe 4x L40S 48GB;
- model: `Qwen/Qwen2.5-3B-Instruct`;
- runtime: vLLM 0.9.2, Torch 2.7.0 CUDA 12.6, Transformers 4.51.3;
- tensor parallelism: `TENSOR_PARALLEL_SIZE=4`;
- populations: `1024 4096`;
- screen prompts: `64`;
- holdout prompts: `256`;
- promoted candidates: `64`;
- adapter bench: `BENCH_ADAPTERS=8`;
- staged search: skipped with `RUN_HALVING=0`.

Validation command:

```bash
python -m optimus.cli validate-run \
  --root results/prime_runs/l40sx4_20260523_2237/results/optimus_gpu_suite_v092_noflash_tp4 \
  --systems-out results/prime_runs/l40sx4_20260523_2237/results/report/optimus_systems_v092_noflash_tp4 \
  --populations 1024,4096 \
  --bench-adapters 8 \
  --skip-halving \
  --strict
```

This passed, and the report PNGs passed `file` checks as valid PNG images.
The committed public report bundle is `docs/reports/l40sx4_20260523_2237/`.

Quality summary:

| run | base holdout | screen-selected holdout | screen-selected delta | promoted holdout oracle | oracle delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| P1024 | `18/256` | `26/256` | `+8/256` | `38/256` | `+20/256` |
| P4096 | `18/256` | `28/256` | `+10/256` | `38/256` | `+20/256` |

Both full-search screen winners transferred positively on the 4x run. Treat the
screen-selected column as the selector evidence and the promoted holdout-oracle
column as candidate-generation evidence; the oracle column is not a selector
claim.

Remaining publication-grade gaps:

- rerun on the intended 8xA100-class target when provider inventory is usable,
  if larger systems evidence is needed beyond the accepted 4x fallback;
- run staged P1024 search for prompt-eval savings and selected-regret plots;
- add trusted HF/PEFT or dense-reference confirmation for promoted candidates;
- broaden selector confirmation beyond this single 4x panel.

The authoritative pod ledger is `.opencode/prime-gpu-ledger.md`.

## Prime GPU Discipline

Before launching on Prime, create or update `.opencode/prime-gpu-ledger.md` with:

- pod id and provider region;
- GPU type/count;
- launch time;
- run command;
- expected shutdown condition.

At the end of the run, record the shutdown time and verify no active pods remain.
