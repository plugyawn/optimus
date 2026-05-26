# Triton Counter Expand v1

Date: 2026-05-26

Hardware: Prime Intellect L40S 48GB pod `a779d03a573a4bb18ecd794e689b0ec1`

## Change

Added a kernel-facing random-field law, `counter_gaussian_v1`, plus
`OPTIMUS_LAZY_DELTA_BACKEND=triton-counter`. The backend requires
`counter_gaussian_v1` candidates and generates `G_t,c[j,a]` inside the Triton
expand kernel from stable candidate seed, sign, target hash, output index, and
basis index.

This removes materialized `B` stacks from the expand stage. It does not yet
fuse the `Qx` shrink or integrate as a vLLM custom op.

## Validation

Remote focused tests:

```bash
PYTHONPATH=. pytest tests/test_vllm_lazy_hook.py -q
```

Result: `31 passed in 2.97s`.

Standalone counter microbench:

- shape: rows=37, rank=16, output_dim=96, candidates=5
- correctness: exact against CPU `counter_gaussian_v1`, max diff `0.0`
- latency: `0.0270 ms` mean over 100 iterations

Expand A/B, warmup excluded:

| rows | rank | output_dim | candidates | materialized B ms | stateless counter ms | ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 64 | 1024 | 16 | 0.0465 | 0.0500 | 1.08x |
| 256 | 64 | 1024 | 16 | 0.1850 | 0.0935 | 0.51x |
| 256 | 128 | 4096 | 16 | 1.2810 | 0.4008 | 0.31x |
| 512 | 128 | 4096 | 16 | 2.7291 | 0.7207 | 0.26x |

Artifacts:

- `results/remote_lazy_kernel_validation/l40s_counterkernel/counter_kernel_probe/microbench_summary.json`
- `results/remote_lazy_kernel_validation/l40s_counterkernel/counter_kernel_probe/expand_ab_summary.json`
- `results/remote_lazy_kernel_validation/l40s_counterkernel/counter_kernel_probe/expand_ab_latency.png`

## Interpretation

This is the first positive evidence that the production lever is the fused
kernel path rather than more adapter or Python-hook work. For small shapes the
stateless kernel is about parity with materialized expand. For larger output
and rank, it is materially faster because it avoids loading candidate B-stack
tiles from memory.

The remaining work is to fuse or custom-op the full lazy delta path:

```text
z = Qx
delta = beta * G(seed, target, row, rank) z
y += delta
```

The current backend still computes `z = Qx` separately and still runs through
Python hook scheduling, so it should not be treated as final p128/p1024
throughput evidence.

## Follow-up End-to-End Gate

The first p128 end-to-end run exposed a specialization bug: the Triton kernels
treated `rows` as a constexpr, so decode-time row-count variation caused
excessive specialization churn. After changing `rows` to a runtime scalar:

| run | backend | population/prompts | candidate batch | result |
| --- | --- | ---: | ---: | ---: |
| p16 warm | triton-counter | p16/4 | 16 | `11.112 cand/s` |
| p128 warm | triton-counter | p128/8 | 16 | `4.997 cand/s` |
| p128 warm | vLLM LoRA-kernel bridge | p128/8 | 16 | `6.110 cand/s` |
| p128 warm | triton-counter | p128/8 | 32 | `12.422 cand/s` |
| p128 warm | vLLM LoRA-kernel bridge | p128/8 | 32 | OOM on L40S |
| p128 warm | triton-counter | p128/8 | 64 | `12.225 cand/s` |
| p1024 warm | triton-counter | p1024/8 | 32 | `12.234 cand/s` |

The current best measured setting is cbs32. cbs64 reduces kernel time but does
not improve overall candidate/sec on this workload.

Artifacts:

- `results/remote_lazy_kernel_validation/l40s_counterp128/counter_p128/p128_row_runtime_cbs32/`
- `results/remote_lazy_kernel_validation/l40s_counterp128/counter_p128/p1024_row_runtime_cbs32/`
- `results/remote_lazy_kernel_validation/l40s_counterp128/counter_p128/counter_end_to_end_throughput.png`
