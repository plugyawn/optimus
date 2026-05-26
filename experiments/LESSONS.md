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
- p128 throughput tests on 48GB L40S need a scheduling cap. The p128/32-prompt
  run can OOM after many candidate chunks because vLLM retains or fragments
  enough KV/cache state that the measurement stops being about lazy-delta
  throughput. Use the documented p128/8 prompt shape for kernel A/B on L40S,
  or move the full prompt-count gate to an 80GB card.
