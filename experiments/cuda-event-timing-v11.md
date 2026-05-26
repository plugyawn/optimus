# cuda-event-timing-v11

## Change

Added `OPTIMUS_LAZY_TIMING_MODE={host,sync,cuda-events}` to the vLLM lazy hook
path and flushed CUDA-event timing after `llm.generate`.

The purpose is not to optimize the kernel. The purpose is to stop throughput
gates from paying one `torch.cuda.synchronize()` per lazy-hook phase while
still keeping useful device-side timing attribution.

## Validation

Hardware: Prime L40S 48GB, Torch `2.10.0`, vLLM `0.19.0`, Triton `3.6.0`,
FlashInfer `0.6.6`.

Commands:

```bash
PYTHONPATH=. pytest \
  tests/test_vllm_lazy_hook.py \
  tests/test_vllm_subspace_parity_probe.py \
  tests/test_subspace_delta_kernel_bench.py -q

PYTHONPATH=. python scripts/eval_vllm_lazy_k1.py \
  --source-run results/remote_lazy_kernel_validation/l40s_qvpack_p128/p128_cbs64_warm \
  --candidate-id-file results/remote_lazy_kernel_validation/l40s_qvpack_p128/source_p128_cbs32_warm_candidate_ids.txt \
  --out results/remote_lazy_kernel_validation/l40s_eventtiming/p128_cbs64_HOST_OR_EVENT \
  --data data/countdown_generated_1200_seed20260507.json \
  --model Qwen/Qwen3-4B \
  --prompts 8 \
  --seed 2 \
  --effective-rank 64 \
  --targets q_proj,v_proj \
  --scale-multiplier 1.0 \
  --candidate-batch-size 64 \
  --prompt-batch-size 0 \
  --prompt-input text \
  --prompt-variants tight \
  --no-use-chat-template \
  --stop-at-answer \
  --max-new-tokens 32 \
  --gpu-memory-utilization 0.88 \
  --max-model-len 2048 \
  --max-num-batched-tokens 4096 \
  --enforce-eager
```

Result:

- CUDA focused suite: `54 passed`.
- PNG headers validated for all plots.
- Score parity: `128/128` common candidates match exactly.
- Prompt parity: `1024/1024` candidate-prompt rows match exactly.

## Timing

| mode | candidates/sec | output tokens/sec | lazy delta s | dispatch s | kernel s | stack s | Qx s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sync | `6.236` | `1359` | `13.988` | `13.988` | `13.663` | `0.087` | `0.176` |
| cuda-events | `16.452` | `3587` | `2.686` | `1.406` | `2.540` | `0.089` | `0.058` |
| host | `16.576` | `3614` | `1.108` | `1.108` | `0.888` | `0.081` | `0.103` |

Artifacts:

- `results/remote_lazy_kernel_validation/l40s_eventtiming/p128_cbs64_sync/summary.json`
- `results/remote_lazy_kernel_validation/l40s_eventtiming/p128_cbs64_events/summary.json`
- `results/remote_lazy_kernel_validation/l40s_eventtiming/p128_cbs64_host/summary.json`
- `results/remote_lazy_kernel_validation/l40s_eventtiming/plots/throughput.png`
- `results/remote_lazy_kernel_validation/l40s_eventtiming/plots/lazy_timing_breakdown.png`
- `results/remote_lazy_kernel_validation/l40s_eventtiming/plots/validation_summary.md`

## Conclusion

Sync timing was contaminating the p128 throughput gate. The same `Qx + packed
q/v counter add` path moves from `6.24 cand/s` in sync mode to `16.45-16.58
cand/s` in event/host modes with exact score and prompt parity.

Future max-throughput gates should use `OPTIMUS_LAZY_TIMING_MODE=host`.
Attribution gates should use `OPTIMUS_LAZY_TIMING_MODE=cuda-events`. Sync mode
is now only a diagnostic for suspicious asynchronous behavior.

This does not remove the need for the production row-block/custom-op path. It
changes the Amdahl read: the current Qwen p128 packed-counter path is already
near the prior p1024 packed-qv throughput once measurement syncs are removed,
so further work should stay on first-class vLLM execution of one `Qx` per
activation-site row block plus scheduled counter add, not standalone
output-tiled fusion.
