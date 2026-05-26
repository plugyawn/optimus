# Packed q/v Counter Kernel v7

Date: 2026-05-26

Hardware: Prime L40S 48GB

## Change

Added `triton_subspace_add_counter_qv_`, a packed stateless counter add kernel
for fused qkv outputs. For the common q/v target preset, it updates the q and v
slices of the fused qkv output in one launch and leaves k unchanged. The vLLM
lazy hook uses it when:

- `OPTIMUS_LAZY_DELTA_BACKEND=triton-counter-inplace`
- `OPTIMUS_LAZY_QKV_KERNEL_POLICY=packed-qkv`
- the fused target has exactly `q_proj` and `v_proj`

The kernel accepts explicit q/v target hashes and field offsets, so it covers
both `target-split` and `fused-qkv-exact` random-field policies.

## Validation

Local:

```bash
PYTHONPATH=. python -m py_compile \
  optimus/kernels/subspace_delta.py \
  optimus/backends/vllm_lazy_hook.py \
  scripts/bench_subspace_delta_kernels.py

PYTHONPATH=. pytest tests/test_vllm_lazy_hook.py tests/test_subspace_delta_kernel_bench.py -q
```

Result: `33 passed, 7 skipped, 1 warning`.

Remote L40S:

```bash
PYTHONPATH=. pytest tests/test_vllm_lazy_hook.py tests/test_subspace_delta_kernel_bench.py -q
```

Result: `40 passed in 5.21s`.

The p16, p128, and p1024 vLLM split-launch and packed-qkv replay runs have
exact `candidate_scores.jsonl` and `per_prompt.jsonl` parity after dropping
timing fields.

## Kernel A/B

Artifacts:

- `results/remote_lazy_kernel_validation/l40s_qvpack/fp32/kernel_ablation_summary.json`
- `results/remote_lazy_kernel_validation/l40s_qvpack/bf16/kernel_ablation_summary.json`
- `results/remote_lazy_kernel_validation/l40s_qvpack/fp32/kernel_ablation_qv_speedup.png`
- `results/remote_lazy_kernel_validation/l40s_qvpack/bf16/kernel_ablation_qv_speedup.png`

| dtype | rows | rank | q dim | kv dim | split q/v ms | packed q/v ms | speedup | max diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fp32 | 64 | 64 | 1024 | 256 | `0.0481` | `0.0306` | `1.57x` | `0.0` |
| fp32 | 256 | 64 | 1024 | 256 | `0.0805` | `0.0666` | `1.21x` | `0.0` |
| fp32 | 256 | 128 | 4096 | 1024 | `0.4941` | `0.4773` | `1.04x` | `0.0` |
| fp32 | 512 | 128 | 4096 | 1024 | `0.9360` | `0.9274` | `1.01x` | `0.0` |
| bf16 | 64 | 64 | 1024 | 256 | `0.0508` | `0.0295` | `1.72x` | `0.0` |
| bf16 | 256 | 64 | 1024 | 256 | `0.0802` | `0.0654` | `1.23x` | `0.0` |
| bf16 | 256 | 128 | 4096 | 1024 | `0.4957` | `0.4824` | `1.03x` | `0.0` |
| bf16 | 512 | 128 | 4096 | 1024 | `0.9462` | `0.9307` | `1.02x` | `0.0` |

## vLLM Replay A/B

Artifacts:

- `results/remote_lazy_kernel_validation/l40s_qvpack/vllm_p16_split-launches/summary.json`
- `results/remote_lazy_kernel_validation/l40s_qvpack/vllm_p16_packed-qkv/summary.json`
- `results/remote_lazy_kernel_validation/l40s_qvpack_p128/replay_p128_split_cbs64/summary.json`
- `results/remote_lazy_kernel_validation/l40s_qvpack_p128/replay_p128_packed_cbs64/summary.json`
- `results/remote_lazy_kernel_validation/l40s_qvpack_p128/replay_p1024_split_cbs64/summary.json`
- `results/remote_lazy_kernel_validation/l40s_qvpack_p128/replay_p1024_packed_cbs64/summary.json`
- `results/remote_lazy_kernel_validation/l40s_qvpack_p128/plots_packed_replay/throughput.png`
- `results/remote_lazy_kernel_validation/l40s_qvpack_p128/plots_packed_replay/lazy_timing_breakdown.png`

| qkv policy | candidate replay sec | lazy delta s | lazy kernel s | stack s | Qx s |
| --- | ---: | ---: | ---: | ---: | ---: |
| split launches | `1.755` | `26.737` | `26.559` | `0.087` | `0.052` |
| packed q/v | `0.868` | `12.680` | `12.560` | `0.048` | `0.050` |

Fixed-basis p128/p1024 replay:

| population | qkv policy | candidates/sec | candidate replay sec | lazy delta s | lazy kernel s | stack s | Qx s | parity |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 128 | split launches | `4.511` | `0.222` | `22.466` | `21.892` | `0.405` | `0.171` | reference |
| 128 | packed q/v | `7.195` | `0.139` | `11.457` | `11.158` | `0.162` | `0.131` | exact |
| 1024 | split launches | `15.924` | `0.063` | `10.388` | `5.807` | `3.343` | `1.627` | reference |
| 1024 | packed q/v | `16.034` | `0.062` | `5.466` | `3.125` | `1.321` | `1.066` | exact |

The full-search p128/p1024 warmed packed runs also completed:

| population | candidates/sec | lazy delta s | lazy kernel s | stack s | Qx s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | `15.180` | `2.948` | `2.126` | `0.468` | `0.360` |
| 1024 | `14.884` | `6.778` | `3.943` | `1.697` | `1.352` |

The full-search p1024 row is not a strict split-vs-packed parity comparison,
because regenerating the basis changed the basis hash. Use the fixed-basis
replay rows for parity claims.

## Conclusion

This is a useful launch-fusion step and it directly supports the final kernel
path. It also narrows the remaining Amdahl target: q/v packing is strong in
small-row and p128 replay regimes, and it halves p1024 lazy-delta time, but
p1024 end-to-end candidate throughput barely moves because model rollout and
scheduling dominate. The next production lever is still a vLLM custom-op or
scheduler-integrated path that reduces work around the kernel, not just another
q/v add microkernel.
