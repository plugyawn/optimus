# Kernel Lessons

- Adapter-vs-lazy generation score parity is too indirect for arithmetic
  correctness. Current p128 score diffs are symmetric, mostly `1/128`, and max
  `3/128`; use one-step signature/logprob parity to separate math bugs from
  greedy decode sensitivity.
- The vLLM LoRA-kernel bridge is now fast enough to be a baseline. Further
  work should move to a fused delta op; more Python-hook polish is unlikely to
  change the production conclusion.
- The measured fused-kernel opportunity on p128 rank64 is inside lazy delta
  time: `49.34s` out of `97.71s`, with `20.56s` spent in the two vLLM LoRA
  kernels and another `18.36s` in metadata plus factor-stack handling.
- Zero-scale adapter-vs-lazy signature parity passes exactly, while nonzero
  scale still has logprob gaps. That makes the next useful work semantic
  parity at the injection point plus the fused delta op, not more end-to-end
  generation-score probing.
- The field-policy sweep rules out the easy qkv-packing explanation. Native
  vLLM loads the expected q/v GQ tensors exactly, lazy layer deltas match native
  adapter deltas at the injection site to bf16 tolerance, and K remains zero.
  The remaining gap is accumulated kernel-order drift plus bridge overhead; the
  next implementation should go to the fused lazy delta/random-field kernel.
- The first Triton cached-field backend is worth keeping only as a scaffold.
  It proves the row-candidate expand contract exactly on CUDA and removes the
  vLLM LoRA metadata path, but p128 shows the scalar-rank expand plus
  materialized `B` stacks are not the production shape. The next real speed
  attempt must fuse deterministic random-field generation/application and
  eliminate per-target `B` stack construction from the hot path.
- `counter_gaussian_v1` is the right kernel-facing random-field law for the
  next implementation stage. It gives deterministic replay from candidate id
  metadata without SHA256 or `torch.Generator` state in the hot path, and the
  L40S expand A/B shows that generating the field in-kernel can beat loading
  materialized `B` stacks at realistic output/rank sizes. This does not close
  the job: `Qx` is still a separate matmul and Python hook scheduling is still
  outside the kernel.
- Kernel specialization keys matter as much as arithmetic. Marking `rows` as
  `tl.constexpr` made p128 collapse from a warmed p16 `8.35 cand/s` to
  `0.37 cand/s`, because vLLM decode produces many row-count shapes. Keeping
  `rows` as a runtime scalar raised the best p128 result to `12.42 cand/s`.
- The stateless counter backend now has a viable end-to-end operating point:
  p128/8 cbs32 reaches `12.42 cand/s`, and p1024/8 cbs32 holds
  `12.23 cand/s` on L40S. Same-host vLLM LoRA-kernel bridge at cbs16 reaches
  `6.11 cand/s`, while cbs32 OOMs. The remaining weakness is kernel-phase
  efficiency: at cbs16 the bridge kernel is still faster, so the next kernel
  work should target RNG/delta throughput and lower launch/hook overhead rather
  than B-stack handling.
- p128 throughput tests on 48GB L40S need a scheduling cap. The p128/32-prompt
  run can OOM after many candidate chunks because vLLM retains or fragments
  enough KV/cache state that the measurement stops being about lazy-delta
  throughput. Use the documented p128/8 prompt shape for kernel A/B on L40S,
  or move the full prompt-count gate to an 80GB card.
- In-place counter add is a useful but limited fused-kernel staging point.
  Post-fix CUDA hook tests pass `35/35`, and synthetic L40S A/B is exact, but
  the isolated in-place add only wins strongly on small output shapes
  (`1.52x` at rows=64/rank=64/out=1024) and is neutral at larger rank/output
  shapes (`0.98-1.01x` at rank=128/out=4096). The production lever is not
  further Python hook polish or only replacing `main + delta`; it is fusing or
  scheduling `Qx`, counter expansion, and output addition inside vLLM routing.
- The reproducible A6000 ablation confirms the same Amdahl shape in both fp32
  and bf16. Total `Qx + in-place add` speedup is `1.02-1.03x` at
  rank=128/output=4096, while bf16 numerical differences are just add-order
  rounding (`max=0.5`, mean about `0.025`). Naive single-kernel fusion that
  recomputes `Qx` per output tile is therefore unlikely to be the right next
  step; the useful production path is vLLM scheduling/custom-op integration
  that computes one `Qx` per activation site and applies counter expand/add
  without Python per-target overhead.
- Staged A6000 run-level parity separates kernel correctness from adapter
  replay semantics. Zero-scale adapter-vs-lazy parity is exact, but nonzero
  target-split replay still fails strict signature parity even for torch
  materialized lazy and vLLM-LoRA-kernel-in-hook backends. The in-place counter
  kernel is not the culprit; the remaining gap is hook-vs-vLLM-adapter
  injection semantics plus accumulated kernel-order drift. Treat vLLM adapter
  replay as an integration reference, not the final production target for the
  lazy kernel.
- Adapter export must clone repeated activation bases before writing
  safetensors. q/v share the same activation-site `Q`; when tensor dtype is
  already float32, `to(...).contiguous()` can preserve storage aliasing and
  `safetensors` rejects the file. The bridge now clones LoRA tensors after dtype
  conversion.
- Target-output capture confirms the nonzero strict adapter mismatch is
  accumulated execution drift, not an obvious counter-kernel arithmetic bug.
  On A6000 c1/p1, layer-0 target drift is tiny, while worst drift appears at
  layers 33-35; both `triton-counter-inplace` and
  `vllm-lora-kernel`-in-hook fail top-logprob parity around `0.5`. The next
  lever remains vLLM custom-op/scheduling integration for `Qx + counter expand
  + add`; adapter replay is an integration reference, not the production
  equality contract.
- Packed q/v launch fusion is worth keeping, but it does not change the final
  Amdahl conclusion. On L40S p16 replay, packed q/v cuts measured lazy kernel
  time from `26.56s` to `12.56s` with exact non-timing replay parity against
  split launches. The isolated synthetic win is `1.57-1.72x` at small
  rows/rank/output and only `1.01-1.03x` at rank128/output4096, so the next
  production step is still a vLLM custom-op/scheduling path that computes one
  `Qx` per activation site and applies packed counter add without Python
  per-target overhead.
