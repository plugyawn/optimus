# Prime GPU Runbook

This runbook is for launching Optimus on Prime GPU pods without losing cleanup
discipline.

## Preflight

1. Verify there are no unexpected active pods:

```bash
set -a; . /Users/progyan/rando/env.local; set +a
PRIME_DISABLE_VERSION_CHECK=1 prime pods list
```

2. Check availability:

```bash
PRIME_DISABLE_VERSION_CHECK=1 prime availability list --gpu-type A100_40GB --gpu-count 8 --output json
PRIME_DISABLE_VERSION_CHECK=1 prime availability list --gpu-type GH200_96GB --gpu-count 1 --output json
```

3. Record the chosen resource in `.opencode/prime-gpu-ledger.md` before running
work.

## Recommended Sequence

Use a cheap smoke pod first if the image or vLLM environment is uncertain:

```bash
MODE=smoke \
SSH_TARGET=root@POD_HOST \
REMOTE_ROOT=optimus \
TENSOR_PARALLEL_SIZE=1 \
scripts/prime_sync_and_run.sh
```

Then run the GPU suite workload on 8xA100:

```bash
MODE=gpu-suite \
SSH_TARGET=root@POD_HOST \
REMOTE_ROOT=optimus \
TENSOR_PARALLEL_SIZE=8 \
POPULATIONS="1024 4096" \
scripts/prime_sync_and_run.sh
```

If 8xA100 provisioning is stale or cannot reach SSH, a 4x fallback can still
run the P1024/P4096 suite with tensor parallel size 4:

```bash
MODE=gpu-suite \
SSH_TARGET=root@POD_HOST \
REMOTE_ROOT=optimus \
TENSOR_PARALLEL_SIZE=4 \
POPULATIONS="1024 4096" \
scripts/prime_sync_and_run.sh
```

## Expected Outputs

After the full run, copy back:

```text
results/optimus_gpu_suite/
results/report/optimus_systems/
```

The sync helper fetches remote results into `results/prime_runs/results/` by
default. For named runs, keep the fetched directory under a timestamped
`results/prime_runs/<name>/results/` path. Then run locally:

```bash
optimus validate-run \
  --root results/optimus_gpu_suite \
  --systems-out results/report/optimus_systems \
  --strict
```

## Cleanup

Terminate pods created for this run as soon as smoke or GPU suite work
finishes. Update `.opencode/prime-gpu-ledger.md` with the shutdown timestamp and
verify `prime pods list` returns no active Optimus-owned pod.
