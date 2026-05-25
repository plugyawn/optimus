# Optimus GPU Suite

This runbook defines the P1024/P4096 GPU workloads used to validate Optimus
search quality, GPU throughput, and staged-search behavior.

## Required Runs

| run | default output | purpose |
| --- | --- | --- |
| P1024 full search | `results/optimus_gpu_suite/search_p1024` | Matched quality and systems baseline. |
| P4096 full search | `results/optimus_gpu_suite/search_p4096` | Best-of-N and scaling evidence. |
| P1024 staged search | `results/optimus_gpu_suite/staged_p1024` | Staged-search savings and regret when enabled. |
| Backend throughput benches | `results/optimus_gpu_suite/bench_*` | Candidate/sec, prompts/sec, tokens/sec, and backend-scaling data. |
| Systems report | `results/report/optimus_systems` | Backend/method-aware plot inputs and PNGs for candidate/sec, backend throughput, token throughput, best-of-N, quality scaling, and staging tradeoffs. |
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
CANDIDATE_BATCH_SIZE=auto \
TENSOR_PARALLEL_SIZE=1 \
scripts/run_optimus_gpu_suite.sh
```

The default launcher mode is the explicit LoRA baseline:

```text
BACKEND=vllm
METHOD=lora
```

Subspace planning uses the same launcher, but it must use the final subspace
flags rather than adapter-only `rank`/`sigma`/`max_loras` controls:

```bash
METHOD=subspace \
BASIS_RANK=128 \
BASIS_PROMPTS=32 \
TARGET_PRESET=transformer-linears \
SCALE_MODE=relative-output-rms \
RHO_GRID=0.002,0.005,0.01,0.02 \
BUDGET_POLICY=per-block-equal \
BASIS_KIND=activation-svd \
TOP_K_GRID=1,4,8,16 \
RUN_HALVING=0 \
scripts/run_optimus_gpu_suite.sh
```

For a p128/p256/p512/p1024 subspace validation plan:

```bash
optimus run-plan \
  --out results/optimus_gpu_suite_subspace/plan.json \
  --root results/optimus_gpu_suite_subspace \
  --populations 128,256,512,1024 \
  --method subspace \
  --backend vllm \
  --match-screen-to-holdout-base-exact \
  --screen-pool-prompts 512 \
  --basis-prompts 32 \
  --basis-rank 128 \
  --target-preset transformer-linears \
  --scale-mode relative-output-rms \
  --rho-grid 0.002,0.005,0.01,0.02 \
  --budget-policy per-block-equal \
  --basis-kind activation-svd \
  --top-k-grid 1,4,8,16 \
  --skip-halving
```

For Qwen3-4B, prefer data-parallel independent jobs or LightEval
`--data-parallel-size` when using a multi-GPU node; tensor parallelism is mainly
for larger models that do not fit or run efficiently on one GPU.

## Acceptance Gates

The generated evidence is not enough by itself. A final claim requires:

1. For subspace runs: `subspace_state.pt`, `subspace_state_summary.json`,
   `candidates.jsonl`, `candidate_scores.jsonl`, `top_k_ensemble.json`,
   `validation_report.json`, `systems_report.json`, and `summary.json`.
   Legacy LoRA baselines keep their adapter rows and candidate summaries under
   an explicitly LoRA-labelled run.
2. Candidate/sec, prompts/sec, token/sec, load time, and eval elapsed time.
3. Heldout evaluation for promoted candidates.
4. Backend parity or trusted confirmation before using any fast backend ranking
   as the selector of record.
5. P1024/P4096 best-of-N curves from saved candidate summaries.
6. Systems plots from `optimus systems-report`, including backend throughput,
   token throughput, best-of-N, quality scaling, and top-K ensemble quality.
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

- add a final public staged-search route before reintroducing staged P1024
  prompt-eval savings and selected-regret plots;
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
