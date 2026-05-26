# vLLM In-Place Parity Post-Fix

Date: 2026-05-26

Hardware: RTX A6000 48GB, vLLM 0.19.0, torch 2.10.0, FlashInfer 0.6.6, Qwen/Qwen3-4B bf16.

Source run: `results/source_p16_cbs16_warm`, copied locally to
`results/remote_lazy_kernel_validation/a6000_vllm_parity_postfix/`.

## Question

After the target-split q/v offset fix and the in-place counter backend, does
the true lazy hook replay match vLLM subspace-as-LoRA replay at one-token
signature level?

## Commands

The staged probe used:

- adapter reference: `scripts/probe_vllm_subspace_parity.py --mode adapter`
- lazy replay: `scripts/probe_vllm_subspace_parity.py --mode lazy --lazy-delta-backend triton-counter-inplace`
- two candidates, two prompts, `target-split`, rank 64, `rho=0.4`
- prefix caching disabled, `max_model_len=512`, `max_num_batched_tokens=8192`

Focused remote validation:

```bash
PYTHONPATH=. pytest \
  tests/test_vllm_lazy_hook.py \
  tests/test_vllm_subspace_parity_probe.py \
  tests/test_subspace_delta_kernel_bench.py -q
```

Result: `41 passed, 1 warning`.

## Results

| probe | status | generated match | max generated logprob diff | max common top-logprob diff | lazy delta time |
| --- | --- | ---: | ---: | ---: | ---: |
| zero scale, target-split, in-place | pass | 4/4 | 0.0000 | 0.0000 | 0.0000s |
| nonzero, target-split, in-place | fail | 3/4 | 0.0568 | 0.7487 | 0.0126s |
| nonzero, target-split, out-of-place counter | fail | 4/4 | 0.1201 | 0.7487 | 0.0171s |
| nonzero, target-split, torch materialized | fail | 4/4 | 0.0776 | 0.6240 | 45.0167s |
| nonzero, target-split, vLLM-LoRA-kernel-in-hook | fail | 4/4 | 0.0667 | 0.7487 | 45.8340s |
| nonzero, target-split, fp32 adapter/fp32 lazy c1p1 | fail | 1/1 | 0.0467 | 0.3750 | 0.1690s |
| nonzero, fused-qkv-exact c1p1 | fail | 1/1 | 0.0902 | 0.6250 | 21.9536s |

Artifacts:

- `results/remote_lazy_kernel_validation/a6000_vllm_parity_postfix/zero_p2_targetsplit/summary.json`
- `results/remote_lazy_kernel_validation/a6000_vllm_parity_postfix/inplace_p2_targetsplit/summary.json`
- `results/remote_lazy_kernel_validation/a6000_vllm_parity_postfix/counter_outofplace_p2_targetsplit/summary.json`
- `results/remote_lazy_kernel_validation/a6000_vllm_parity_postfix/torch_lazy_p2_targetsplit/summary.json`
- `results/remote_lazy_kernel_validation/a6000_vllm_parity_postfix/vllm_lora_kernel_p2_targetsplit/summary.json`
- `results/remote_lazy_kernel_validation/a6000_vllm_parity_postfix/fp32_c1p1_targetsplit/summary.json`
- `results/remote_lazy_kernel_validation/a6000_vllm_parity_postfix/fused_exact_c1p1/summary.json`

## Conclusion

The in-place counter kernel is not the source of the remaining run-level
strict-parity failure.

Evidence:

1. Zero-scale adapter-vs-lazy parity is exact.
2. Out-of-place counter has the same nonzero failure envelope as in-place.
3. The materialized torch lazy backend also fails nonzero strict adapter parity.
4. The vLLM-LoRA-kernel-in-hook backend also fails nonzero strict adapter parity.
5. Float32 adapter/lazy compute improves but does not close the gap.
6. `fused-qkv-exact` does not close the gap and adds a large first-use Triton
   compile cost on the A6000 probe.

The live conclusion is therefore narrower than "full vLLM adapter parity":
the lazy counter path has exact zero-delta behavior and fast nonzero execution,
but strict nonzero signature parity to vLLM adapter replay remains blocked by
hook-vs-adapter injection semantics and accumulated kernel-order drift.

## Patch Found During Probe

The float32 adapter diagnostic exposed an adapter export bug: q/v `lora_A`
tensors shared storage when no dtype conversion occurred, causing `safetensors`
to reject the export. `subspace_lora_tensors` now clones tensors after dtype
conversion, and the target-split bridge test asserts q/v `lora_A` storage is
not aliased.

## Next Kernel Lever

Do not spend more time polishing Python hook replay as if it will reach strict
production parity. The useful next implementation is the fused/custom-op vLLM
path:

1. compute one `Qx` per activation site under vLLM scheduling;
2. apply counter random-field expansion and output add in a first-class op;
3. carry explicit row-candidate routing metadata;
4. validate against the torch reference at the injection point, then run
   end-to-end score/signature probes as integration tests.

