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

The p16 vLLM split-launch and packed-qkv runs have exact
`candidate_scores.jsonl` and `per_prompt.jsonl` parity after dropping timing
fields.

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

| qkv policy | candidate replay sec | lazy delta s | lazy kernel s | stack s | Qx s |
| --- | ---: | ---: | ---: | ---: | ---: |
| split launches | `1.755` | `26.737` | `26.559` | `0.087` | `0.052` |
| packed q/v | `0.868` | `12.680` | `12.560` | `0.048` | `0.050` |

## Conclusion

This is a useful launch-fusion step and it directly supports the final kernel
path. It also narrows the remaining Amdahl target: q/v packing is strong in
small-row regimes but nearly neutral at larger rank/output shapes. The next
production lever is still a vLLM custom-op/scheduling path that computes one
`Qx` per activation site and applies packed counter add without Python
per-target hook overhead.
