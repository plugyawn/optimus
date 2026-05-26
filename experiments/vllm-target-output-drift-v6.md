# vLLM Target-Output Drift Capture v6

Date: 2026-05-26

Hardware: Prime A6000 48GB

Model/runtime:

- `Qwen/Qwen3-4B`
- Torch `2.10.0+cu128`
- vLLM `0.19.0`
- FlashInfer `0.6.6`
- Triton `3.6.0`

## Question

Strict nonzero vLLM adapter-vs-lazy signature parity still fails after the
in-place counter target-split fix. This experiment asks whether the remaining
gap is caused by the stateless counter in-place kernel, or by accumulated
hook-vs-adapter execution semantics.

## Validation

Local:

```bash
PYTHONPATH=. pytest tests/test_vllm_subspace_parity_probe.py \
  tests/test_vllm_lazy_hook.py::test_subspace_adapter_bridge_target_split_uses_requested_targets_only -q
PYTHONPATH=. python -m py_compile scripts/probe_vllm_subspace_parity.py
git diff --check
```

Remote:

```bash
PYTHONPATH=. pytest tests/test_vllm_subspace_parity_probe.py \
  tests/test_vllm_lazy_hook.py::test_subspace_adapter_bridge_target_split_uses_requested_targets_only -q
```

Result: `6 passed` locally and remotely.

## Results

One candidate and one prompt were captured with `--capture-target-outputs` over
all q/v-packed attention targets. Native vLLM adapter replay produced the
reference target outputs; lazy replay was run twice.

| lazy backend | status | generated match | max common top-logprob diff | target max abs | target RMS mean | worst layer | lazy delta s | kernel s | stack s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `triton-counter-inplace` | fail | 1/1 | `0.5011` | `1.5` | `0.0216` | 35 | `18.46` | `18.44` | `0.01` |
| `vllm-lora-kernel` hook | fail | 1/1 | `0.4991` | `0.75` | `0.0168` | 33 | `38.24` | `0.04` | `38.11` |

All 36 captured target-output comparisons had matching shapes and no missing
rows. For the in-place counter backend, layer 0 target-output drift was only
`0.00195` max abs; the visible drift accumulates in late layers.

## Artifacts

- `results/remote_lazy_kernel_validation/a6000_drift_capture/target_output_drift_by_layer.png`
- `results/remote_lazy_kernel_validation/a6000_drift_capture/target_output_capture_summary.json`
- `results/remote_lazy_kernel_validation/a6000_drift_capture/c1p1_targetsplit/summary.json`
- `results/remote_lazy_kernel_validation/a6000_drift_capture/c1p1_targetsplit/target_output_drift.csv`
- `results/remote_lazy_kernel_validation/a6000_drift_capture/c1p1_vllm_lora_kernel/summary.json`
- `results/remote_lazy_kernel_validation/a6000_drift_capture/c1p1_vllm_lora_kernel/target_output_drift.csv`

## Conclusion

The new capture rules out the simple explanation that strict nonzero parity is
broken by the in-place counter kernel alone. The same failure class appears when
the lazy hook calls vLLM's LoRA kernels, and the mismatch grows across layers.
The remaining gap is best treated as adapter replay versus hook execution
semantics plus accumulated bf16/order drift.

The production path should still go to fused/custom-op vLLM integration. The
counter backend removes per-candidate factor-stack construction, while the
vLLM-LoRA-kernel hook spends the measured c1/p1 lazy time almost entirely in
stack setup. Further Python-hook emulation is unlikely to be the final lever.
