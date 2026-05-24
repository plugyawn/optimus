# Optimus GPU Suite

This runbook defines the P1024/P4096 GPU workloads used to validate Optimus
search quality, GPU throughput, and staged-search behavior.

## Required Runs

| run | default output | purpose |
| --- | --- | --- |
| P1024 full search | `results/optimus_gpu_suite/search_p1024_chunk32` | Matched quality and systems baseline. |
| P4096 full search | `results/optimus_gpu_suite/search_p4096_chunk32` | Best-of-N and scaling evidence. |
| P1024 halving | `results/optimus_gpu_suite/halving_p1024_stage8_surv64` | Staged-search savings and regret. |
| Adapter throughput benches | `results/optimus_gpu_suite/bench_a*_p64` | Candidate/sec, prompts/sec, tokens/sec, adapter-scaling data. |
| Systems report | `results/report/optimus_systems` | Backend/method-aware plot inputs and PNGs for candidate/sec, adapter throughput, token throughput, best-of-N, quality scaling, and staging tradeoffs. |
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
MODEL=Qwen/Qwen3-4B \
PROMPTS=64 \
HOLDOUT_PROMPTS=256 \
PROMOTE=64 \
CHUNK_ADAPTERS=32 \
MAX_LORAS=32 \
MAX_CPU_LORAS=8192 \
TENSOR_PARALLEL_SIZE=1 \
scripts/run_optimus_gpu_suite.sh
```

For Qwen3-4B, prefer data-parallel independent jobs or LightEval
`--data-parallel-size` when using a multi-GPU node; tensor parallelism is mainly
for larger models that do not fit or run efficiently on one GPU.

## Acceptance Gates

The generated evidence is not enough by itself. A final claim requires:

1. `summary.json` and per-prompt rows for each full search.
2. Candidate/sec, prompts/sec, token/sec, load time, and eval elapsed time.
3. Heldout evaluation for promoted candidates.
4. Backend parity or trusted confirmation before using any fast backend ranking
   as the selector of record.
5. P1024/P4096 best-of-N curves from saved candidate summaries.
6. Systems plots from `optimus systems-report`, including `adapter_throughput.png`, `token_throughput.png`, `best_of_n.png`, and `quality_scaling.png`.
7. `optimus validate-run` passes for the run root and systems report.
8. Active GPU pods stopped or explicitly reported after the run.

## Validation State

Run artifacts are intentionally local and ignored. Validate a completed suite
with:

```bash
python -m optimus.cli validate-run \
  --root results/optimus_gpu_suite \
  --systems-out results/report/optimus_systems \
  --populations 1024,4096 \
  --strict
```

Reports should present selector and candidate-generation evidence separately.
Treat the screen-selected column as selector evidence and the promoted
holdout-oracle column as candidate-generation evidence; the oracle column is
not a selector claim. Concrete numbers should come from the local generated
`quality_scaling.csv`, not from hand-copied tables in the repository.

Remaining publication-grade gaps:

- run staged P1024 search for prompt-eval savings and selected-regret plots;
- add trusted Transformers, dense-reference, or LightEval confirmation for the
  selected materialized model state;
- broaden selector confirmation across more than one matched panel.

The authoritative pod ledger is `.opencode/prime-gpu-ledger.md`.

## Prime GPU Discipline

Before launching on Prime, create or update `.opencode/prime-gpu-ledger.md` with:

- pod id and provider region;
- GPU type/count;
- launch time;
- run command;
- expected shutdown condition.

At the end of the run, record the shutdown time and verify no active pods remain.
