# row-mapping-cache-v10

## Change

Reused the existing vLLM token-mapping cache for all stateless counter lazy
backends:

- `triton-counter`
- `triton-counter-inplace`
- packed q/v counter add
- guarded packed q/v fused-from-`x`

The runtime now reports:

- `row_mapping_cache_hits`
- `row_mapping_cache_misses`

The cache is controlled by `OPTIMUS_LAZY_ROW_MAPPING_CACHE_SIZE`, defaulting to
`64`. Setting it to `0` disables caching and forces a fresh device
`row_candidate_id` mapping for each delta call.

## Validation

Local:

```bash
PYTHONPATH=. python -m py_compile \
  optimus/backends/vllm_lazy_hook.py \
  scripts/eval_vllm_lazy_k1.py \
  scripts/probe_vllm_subspace_parity.py

PYTHONPATH=. pytest \
  tests/test_vllm_lazy_hook.py \
  tests/test_vllm_subspace_parity_probe.py \
  tests/test_subspace_delta_kernel_bench.py \
  -q
```

Result: `43 passed, 9 skipped`.

Remote A6000:

```bash
PYTHONPATH=. pytest \
  tests/test_vllm_lazy_hook.py \
  tests/test_vllm_subspace_parity_probe.py \
  tests/test_subspace_delta_kernel_bench.py \
  -q
```

Result: `52 passed`.

## Timing

Hardware: Prime A6000 48GB. Model: `Qwen/Qwen3-4B`. Shape:
p128, 8 prompts, rank 64, q/v targets, packed q/v counter add, cbs64,
`--stop-at-answer --max-new-tokens 32`.

| run | row-map cache | candidates/sec | sec/candidate | lazy delta s | kernel s | stack s | Qx s | row-map hits | row-map misses |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `p128_cbs64_cache_on` | on | `7.648` | `0.1308` | `5.990` | `0.607` | `0.163` | `0.398` | `4760` | `136` |
| `p128_cbs64_cache_off` | off | `7.369` | `0.1357` | `6.188` | `0.628` | `0.321` | `0.431` | `0` | `4896` |

Artifacts:

- `results/remote_lazy_kernel_validation/a6000_rowmap/p128_cbs64_cache_on/summary.json`
- `results/remote_lazy_kernel_validation/a6000_rowmap/p128_cbs64_cache_off/summary.json`
- `results/remote_lazy_kernel_validation/a6000_rowmap/p128_cbs64_stop32/summary.json`
- `results/remote_lazy_kernel_validation/a6000_rowmap/p128_cbs64_stop32_warm/summary.json`

## Conclusion

The cache is correct and worth keeping. It removes repeated row-candidate
metadata copies for stable vLLM decode shapes, cuts measured stack time roughly
in half on this p128 gate, and gives a small throughput improvement.

This does not change the main systems conclusion. The literal `Qx + counter
add` microkernel already exists and is a measured dead end when it recomputes
`Qx` per output tile. The next useful kernel work is a vLLM row-block or
custom-op path that computes one `Qx` per activation-site row block and applies
packed counter add without Python hook dispatch, adapter materialization, or
repeated metadata movement.
