# Triton Counter In-Place v3

Date: 2026-05-26

Hardware:

- Prime Intellect L40S 48GB pod `331b35e07ab84b1ea402433218bd9c09` for the
  end-to-end p16/p128/p1024 speed gates.
- Prime Intellect L40S 48GB pod `2aab4c16937f4eab8b979fcd0957eaab` for the
  post-fix CUDA ABI/offset validation and synthetic kernel A/B.

## Change

Added `OPTIMUS_LAZY_DELTA_BACKEND=triton-counter-inplace`.

The backend reuses the `counter_gaussian_v1` random field and separate
`z = Qx` shrink, but the Triton expand kernel writes directly into the vLLM
linear output:

```text
y[row, output_offset + j] += beta * G(seed, target, j, r) z[row, r]
```

This removes full-delta allocation, q/v slice copy, and the final PyTorch
`main + delta` from the forward hook path. It is not yet the final vLLM custom
operator because hook dispatch and separate `Qx` remain.

## Validation

Remote focused CUDA tests:

```bash
PYTHONPATH=. pytest tests/test_vllm_lazy_hook.py -q
```

Initial result on the speed pod: `34 passed in 7.62s`.

Post-fix result after adding the target-split q/v field-offset regression test:
`35 passed in 6.52s`.

The post-fix CUDA run caught and closed two launch ABI bugs:

1. the out-of-place counter kernel accepted `field_output_offset` but the
   launcher did not pass it;
2. the in-place counter kernel launcher passed `field_output_offset` but the
   Triton kernel signature did not accept it.

It also validates that target-split fused-qkv fields use local q/v random-field
indices while writing into the fused qkv output slice.

## Speed

| run | candidate batch | candidates/sec | lazy delta s | kernel s | stack s | Qx s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cold p16 | 16 | `0.801` | `29.083` | `28.761` | `0.165` | `0.129` |
| warm p16 | 16 | `11.434` | `1.053` | `0.740` | `0.164` | `0.129` |
| warm p128 | 32 | `13.647` | `4.269` | `2.851` | `0.787` | `0.578` |
| warm p128 | 64 | `13.924` | `4.273` | `2.818` | `0.821` | `0.578` |
| warm p1024 | 64 | `13.808` | `10.559` | `5.756` | `2.744` | `1.940` |

Previous best out-of-place counter references:

| run | candidate batch | candidates/sec |
| --- | ---: | ---: |
| p128 | 32 | `12.422` |
| p128 | 64 | `12.225` |
| p1024 | 32 | `12.234` |

The in-place backend is a real end-to-end win, but it is a modest one:
roughly `13-14%` on p128/p1024. That matches the Amdahl expectation that the
final add/allocation is one bottleneck, not the whole lazy path.

## Synthetic kernel A/B

Command shape: one warmed Python process on L40S, fp32 tensors,
`counter_gaussian_v1`, 16 candidates, 200 timed iterations per row. The
out-of-place path measures `triton_subspace_expand_counter(...)` followed by
`base + delta`; the in-place path measures
`triton_subspace_add_counter_(..., output)`.

| rows | rank | output dim | max diff | expand+add ms | in-place add ms | speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 64 | 1024 | `0.0` | `0.0463` | `0.0304` | `1.52x` |
| 256 | 64 | 1024 | `0.0` | `0.0523` | `0.0491` | `1.06x` |
| 256 | 128 | 4096 | `0.0` | `0.3672` | `0.3763` | `0.98x` |
| 512 | 128 | 4096 | `0.0` | `0.7196` | `0.7156` | `1.01x` |

The narrow in-place add lever helps small output shapes, but it is basically
neutral at the larger rank/output shapes. That makes the next production lever
more specific: fuse or schedule `Qx`, counter expansion, and output addition
inside the vLLM execution path instead of only replacing the final add.

## Replay

Internal in-place replay:

| comparison | common candidates | score mismatches | prompt exact match | text match |
| --- | ---: | ---: | ---: | ---: |
| p128 cbs64 vs p128 cbs32 | 128 | 0 | `1.0000` | `0.8818` |
| p128 cbs64 vs p1024 cbs64 prefix | 128 | 0 | `1.0000` | `1.0000` |

Against the old out-of-place p128 cbs32 reference:

| comparison | common candidates | score mismatches | max score diff | prompt exact match | text match |
| --- | ---: | ---: | ---: | ---: | ---: |
| old p128 cbs32 vs in-place p128 cbs64 | 128 | 2 | `0.125` | `0.9980` | `0.3398` |
| old p128 cbs32 vs in-place p1024 cbs64 prefix | 128 | 2 | `0.125` | `0.9980` | `0.3398` |

So the in-place backend is internally stable across scale. The old
out-of-place-vs-in-place replay comparison predates the target-split q/v
field-offset fix, so it should be treated as historical systems evidence, not
as the final strict replay comparison.

## Artifacts

- `results/remote_lazy_kernel_validation/l40s_counter_inplace/inplace_counter/p128_cbs64/`
- `results/remote_lazy_kernel_validation/l40s_counter_inplace/inplace_counter/p1024_cbs64/`
- `results/remote_lazy_kernel_validation/l40s_counter_inplace/plots_inplace_vs_counter/throughput.png`
- `results/remote_lazy_kernel_validation/l40s_counter_inplace/plots_inplace_vs_counter/lazy_timing_breakdown.png`
- `results/remote_lazy_kernel_validation/l40s_counter_inplace/plots_inplace_vs_counter/validation_summary.md`
- `results/remote_lazy_kernel_validation/l40s_counter_inplace/plots_inplace_batch_parity/validation_summary.md`

## Interpretation

This is worth keeping because it moves work into the Triton kernel and improves
the warmed p128/p1024 regime. It does not remove enough overhead to end the
systems work. The production-grade next step is a vLLM-owned custom operator
that keeps row-candidate routing inside the execution path and fuses or
schedules `Qx`, counter expand, and in-place output add without Python
forward-hook overhead. More Python-hook polishing is now lower priority unless
profiling shows the fused/custom-op path is blocked by hook dispatch itself.
