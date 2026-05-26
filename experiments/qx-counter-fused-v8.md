# qx-counter-fused-v8

## Change

Added a guarded prototype for direct fused `Qx + packed q/v counter add`:

- `triton_subspace_add_counter_qv_from_x_`
- `OPTIMUS_LAZY_QKV_KERNEL_POLICY=packed-qkv-from-x`
- benchmark columns and plots comparing:
  - `Qx + split q/v add`
  - `Qx + packed q/v add`
  - direct fused-from-`x` q/v counter add

The prototype is intentionally not the default path.

## Validation

Hardware: Prime A6000 48GB, Torch `2.12.0+cu130`, Triton `3.7.0`.

Commands:

```bash
PYTHONPATH=. pytest tests/test_vllm_lazy_hook.py tests/test_subspace_delta_kernel_bench.py -q
PYTHONPATH=. python scripts/bench_subspace_delta_kernels.py --out results/qx_counter_fused/fp32 --dtype float32 --iters 120 --warmup 10
PYTHONPATH=. python scripts/bench_subspace_delta_kernels.py --out results/qx_counter_fused/bf16 --dtype bfloat16 --iters 120 --warmup 10
```

Results:

- CUDA tests: `42 passed`.
- fp32 fused-from-`x` max diff: `3.8e-05` to `9.9e-05`.
- bf16 fused-from-`x` max diff: `0.5` to `1.0`, matching expected low-precision accumulation/add-order scale.
- PNG headers validated for all fetched plots.

## Timing

| dtype | shape | Qx + packed q/v ms | fused from x ms | fused speed vs packed |
| --- | --- | ---: | ---: | ---: |
| fp32 | rows64_r64_out1024_c16 | `0.122` | `1.884` | `0.065x` |
| fp32 | rows256_r64_out1024_c16 | `0.152` | `5.810` | `0.026x` |
| fp32 | rows256_r128_out4096_c16 | `1.004` | `41.398` | `0.024x` |
| fp32 | rows512_r128_out4096_c16 | `1.958` | `91.148` | `0.021x` |
| bf16 | rows64_r64_out1024_c16 | `0.115` | `1.854` | `0.062x` |
| bf16 | rows256_r64_out1024_c16 | `0.139` | `4.544` | `0.031x` |
| bf16 | rows256_r128_out4096_c16 | `0.990` | `32.478` | `0.030x` |
| bf16 | rows512_r128_out4096_c16 | `1.919` | `62.311` | `0.031x` |

Artifacts:

- `results/remote_lazy_kernel_validation/a6000_qxcounter/fp32/kernel_ablation_qv_fused_from_x.png`
- `results/remote_lazy_kernel_validation/a6000_qxcounter/bf16/kernel_ablation_qv_fused_from_x.png`
- `results/remote_lazy_kernel_validation/a6000_qxcounter/fp32/kernel_ablation_summary.json`
- `results/remote_lazy_kernel_validation/a6000_qxcounter/bf16/kernel_ablation_summary.json`

## Conclusion

The literal output-tiled single-kernel formulation is correct but not viable.
It recomputes local `Qx` for every output tile, so Qwen-shaped
`input_dim=4096` workloads are tens of times slower than the current
`torch Qx + packed q/v counter add` path.

The next production path should not be this kernel. It should compute one
`Qx` per activation-site row block and then apply the counter add without
materialized adapters, likely via vLLM custom-op scheduling or a two-stage
shrink/expand pipeline with row-candidate routing. This experiment rules out
the naive fused-from-`x` kernel as the final lever.
