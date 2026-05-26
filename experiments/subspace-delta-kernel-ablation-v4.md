# Subspace Delta Kernel Ablation v4

Date: 2026-05-26

Hardware: Prime Intellect A6000 48GB pod
`42cabed171834391974c90045e8c49ac`.

## Change

Added `scripts/bench_subspace_delta_kernels.py`, a reproducible CUDA
microbenchmark for the current lazy-delta kernels. The benchmark writes:

- `kernel_ablation_summary.json`
- `kernel_ablation.csv`
- `kernel_ablation_latency.png`
- `kernel_ablation_speedup.png`

The harness measures the stages that matter for the next fused/custom-op
decision:

1. `Qx = x @ Q.T`
2. materialized cached-field expand
3. stateless counter expand
4. stateless counter expand plus PyTorch output add
5. stateless counter in-place output add
6. total `Qx + counter expand + add`
7. total `Qx + counter in-place add`

## Validation

Local:

```bash
python -m py_compile scripts/bench_subspace_delta_kernels.py
PYTHONPATH=. pytest tests/test_subspace_delta_kernel_bench.py -q
PYTHONPATH=. pytest -q
```

Results:

- focused local harness tests: `3 passed`
- full local suite: `236 passed, 5 skipped`

Remote A6000:

```bash
PYTHONPATH=. pytest tests/test_vllm_lazy_hook.py tests/test_subspace_delta_kernel_bench.py -q
```

Result: `38 passed in 7.38s`.

## Artifacts

- `results/remote_lazy_kernel_validation/a6000_kernel_ablation/fp32/kernel_ablation_summary.json`
- `results/remote_lazy_kernel_validation/a6000_kernel_ablation/fp32/kernel_ablation_latency.png`
- `results/remote_lazy_kernel_validation/a6000_kernel_ablation/fp32/kernel_ablation_speedup.png`
- `results/remote_lazy_kernel_validation/a6000_kernel_ablation/bf16/kernel_ablation_summary.json`
- `results/remote_lazy_kernel_validation/a6000_kernel_ablation/bf16/kernel_ablation_latency.png`
- `results/remote_lazy_kernel_validation/a6000_kernel_ablation/bf16/kernel_ablation_speedup.png`

All PNG headers were verified after sync.

## FP32 Results

| rows | rank | output dim | counter expand ms | in-place add ms | total out-of-place ms | total in-place ms | total speedup | max diff |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 64 | 1024 | `0.0438` | `0.0359` | `0.1038` | `0.0842` | `1.232x` | `0.0` |
| 256 | 64 | 1024 | `0.1060` | `0.1064` | `0.1348` | `0.1312` | `1.028x` | `0.0` |
| 256 | 128 | 4096 | `0.7498` | `0.7514` | `0.7965` | `0.7806` | `1.020x` | `0.0` |
| 512 | 128 | 4096 | `1.4825` | `1.4759` | `1.5723` | `1.5343` | `1.025x` | `0.0` |

## BF16 Results

| rows | rank | output dim | counter expand ms | in-place add ms | total out-of-place ms | total in-place ms | total speedup | max diff | mean diff |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 64 | 1024 | `0.0412` | `0.0367` | `0.1053` | `0.0755` | `1.395x` | `0.5` | `0.0180` |
| 256 | 64 | 1024 | `0.1075` | `0.1094` | `0.1201` | `0.1178` | `1.020x` | `0.5` | `0.0179` |
| 256 | 128 | 4096 | `0.7613` | `0.7689` | `0.7912` | `0.7796` | `1.015x` | `0.5` | `0.0255` |
| 512 | 128 | 4096 | `1.5022` | `1.5202` | `1.5533` | `1.5240` | `1.019x` | `0.5` | `0.0253` |

The bf16 diff is expected add-order rounding: the fp32 path is exact, while
bf16 uses a PyTorch out-of-place add for the reference and a Triton in-place
load/add/store for the candidate path.

## Interpretation

This is the cleanest Amdahl answer so far:

- Stateless counter expand remains the right direction versus materialized
  cached fields, especially at larger output/rank shapes.
- In-place add is valuable for small row/output shapes, but the total speedup
  is only about `1.02x` at the rank-128/output-4096 shapes that dominate full
  transformer-linears work.
- `Qx` is a small fraction of large-shape total in this synthetic setting
  (`~1-5%`), so naive single-kernel fusion that recomputes `Qx` per output
  tile is unlikely to help.

The next production lever is therefore not a narrower in-place-add kernel.
It is a vLLM-owned custom path that removes hook dispatch, preserves explicit
row-candidate routing, schedules one `Qx` per activation site, and applies the
counter expand/add without materialized adapters or per-target Python overhead.
