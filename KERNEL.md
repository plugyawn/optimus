# Optimus Lazy Subspace Kernel Contract

## Target Operation

For each patched transformer linear target `t`, activation row `x_n`, and
candidate id `c(n)`, compute:

```text
y_n = W_t x_n + beta_t G_t,c(n) Q_s(t) x_n
```

The production kernel target is the lazy delta term only:

```text
delta_n = beta_t G_t,c(n) Q_s(t) x_n
```

Inputs:

- `x`: `[tokens, input_dim]`, fp16/bf16 activation rows from vLLM.
- `Q`: `[rank, input_dim]`, activation-site basis shared by all candidates at
  the site.
- `G`: deterministic candidate/target random field, logically
  `[candidates, output_dim, rank]`.
- `row_candidate_id`: `[tokens]`, explicit row-to-candidate mapping.
- `beta`: per-target scale.

Output:

- `delta`: `[tokens, output_dim]`, same dtype as the target linear output.

The kernel must not depend on request order, scheduler order, global RNG state,
or vLLM adapter loading/swapping. Prefix caching must be disabled or
candidate-keyed whenever q/k/v perturbations are active.

## Reference Implementations

- Tensor reference: `LazyHookRuntime.delta(..., delta_backend="torch")`.
- Current GPU bridge: `OPTIMUS_LAZY_DELTA_BACKEND=vllm-lora-kernel`, which
  reuses vLLM Triton LoRA shrink/expand kernels with cached `A=Q` and
  `B=beta*G` factor stacks.
- Native stateless expand scaffold:
  `OPTIMUS_LAZY_DELTA_BACKEND=triton-counter`, which requires
  `counter_gaussian_v1` candidates and generates `G_t,c` inside the Triton
  expand kernel instead of materializing `B` stacks.
- Native stateless in-place add scaffold:
  `OPTIMUS_LAZY_DELTA_BACKEND=triton-counter-inplace`, which uses the same
  counter random field but writes `y += beta * G_t,c z` directly into the
  vLLM linear output buffer. It still computes `z = Qx` separately, but avoids
  the full delta allocation, q/v slice assignment, and final PyTorch add.
- Native adapter baseline: `scripts/eval_vllm_subspace_adapter_k1.py`.
- Signature parity probe: `scripts/probe_vllm_subspace_parity.py`.

The bridge is a validated stepping stone, not the final fused kernel. It still
pays Python hook dispatch, separate shrink and expand launches, explicit delta
allocation/add, row-metadata prep, and factor-stack management.

## Correctness Gates

Quick local:

```bash
PYTHONPATH=. pytest tests/test_vllm_lazy_hook.py tests/test_vllm_subspace_parity_probe.py -q
```

Full local:

```bash
PYTHONPATH=. pytest -q
```

GPU parity probe, small:

```bash
PYTHONPATH=. python scripts/probe_vllm_subspace_parity.py \
  --mode adapter \
  --source-run results/remote_vllm_lazy_hook/l40s/rebuilt_basis_p1024_activationsvd_rho0p4_source \
  --candidate-id-file results/remote_vllm_lazy_hook/l40s/p16_candidate_ids.txt \
  --out results/lazy_kernel_parity_probe/p16_signature \
  --data data/countdown_generated_1200_seed20260507.json \
  --model Qwen/Qwen3-4B \
  --prompts 8 \
  --adapter-rank 64 \
  --targets q_proj,v_proj \
  --scale-multiplier 2.0 \
  --prompt-input text \
  --prompt-variants tight \
  --max-logprob-diff 1e-3

PYTHONPATH=. python scripts/probe_vllm_subspace_parity.py \
  --mode lazy \
  --source-run results/remote_vllm_lazy_hook/l40s/rebuilt_basis_p1024_activationsvd_rho0p4_source \
  --candidate-id-file results/remote_vllm_lazy_hook/l40s/p16_candidate_ids.txt \
  --out results/lazy_kernel_parity_probe/p16_signature \
  --data data/countdown_generated_1200_seed20260507.json \
  --model Qwen/Qwen3-4B \
  --prompts 8 \
  --adapter-rank 64 \
  --targets q_proj,v_proj \
  --scale-multiplier 2.0 \
  --prompt-input text \
  --prompt-variants tight \
  --lazy-delta-backend vllm-lora-kernel \
  --max-logprob-diff 1e-3
```

Use staged `adapter` then `lazy` modes for Qwen3-4B-size probes. A single
process `both` run can fail on 48GB GPUs because vLLM may retain GPU memory
after the first LLM instance is destroyed.

Acceptance for arithmetic parity:

- generated one-step token match rate is `1.0`;
- common top-logprob max absolute diff is within the declared dtype tolerance;
- the probe applies nonzero lazy-delta rows;
- candidate ids, rank, targets, dtype, scale, prompt contract, and decode
  contract match between adapter and lazy paths.

Generation-score parity remains secondary because multi-token greedy decoding
can amplify tiny logit differences into ±1 sample score changes.

Current L40S strict signature evidence:

| condition | backend | scale | status | token match | max generated logprob diff | max common top-logprob diff |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| zero scale | torch | 0.0 | pass | 2/2 | 0.0000 | 0.0000 |
| nonzero target-split | torch | 2.0 | fail | 2/2 | 0.0310 | 0.2031 |
| nonzero fused-qkv-exact | torch | 2.0 | fail | 2/2 | 0.0701 | 0.1249 |
| nonzero fused-qkv-exact | vLLM LoRA-kernel bridge, split launches | 2.0 | fail | 2/2 | 0.0287 | 0.1249 |
| nonzero fused-qkv-exact | vLLM LoRA-kernel bridge, packed qkv launch | 2.0 | fail | 2/2 | 0.0797 | 0.1250 |

The zero-scale pass shows the probe is not dominated by run-to-run vLLM noise.
The nonzero failures mean strict adapter-vs-lazy logprob parity is still open,
but the newer field-policy sweep narrows the issue. `fused-qkv-exact` improves
top-logprob agreement versus target-split, and a direct layer probe confirms
native vLLM's loaded q/v adapter tensors match the expected subspace tensors
exactly. On the same layer-0 `qkv_proj` activation, native vLLM adapter delta
and lazy delta agree to max absolute error `0.00586` in bf16, with zero K-slice
leakage. The remaining signature gap is therefore accumulated bf16/kernel-order
drift across layers, not adapter-file generation, qkv packing, or hook
placement.

Do not spend more time on Python-hook variants unless they are a correctness
probe for the fused/custom-op path. The viable next lever is a custom fused
delta/random-field kernel that removes factor-stack construction, vLLM LoRA
metadata prep, and separate shrink/expand launches.

Current L40S cached-field Triton evidence:

| condition | backend | population/prompts | candidates/sec | lazy delta s | kernel s | stack s | result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| microbench | Triton cached-field expand | synthetic | 43,319 it/s | n/a | n/a | n/a | exact vs torch |
| warmed replay | Triton cached-field expand | p16/32 | 3.843 | 2.573 | 0.994 | 1.401 | faster than vLLM bridge |
| replay | vLLM LoRA-kernel bridge | p16/32 | 3.200 | 3.465 | 1.091 | 1.360 | baseline |
| replay | Triton cached-field expand | p128/8 | 6.375 | 14.859 | 5.708 | 8.525 | slower than vLLM bridge |
| replay | vLLM LoRA-kernel bridge | p128/8 | 7.256 | 12.327 | 2.658 | 8.284 | current p128 winner |

The cached-field Triton kernel is therefore a correctness scaffold and an
Amdahl probe, not the production kernel. It removes the vLLM LoRA metadata
path and can win at warmed p16, but p128 shows the naive candidate-routed
expand kernel plus materialized `B` stacks are still too expensive. The next
kernel milestone must generate and apply the deterministic random field inside
the fused op, or otherwise eliminate `B` stack construction and the scalar-rank
expand loop.

Current L40S stateless-counter expand evidence:

| condition | backend | shape | correctness | mean latency |
| --- | --- | --- | --- | ---: |
| microbench | Triton stateless counter expand | rows=37 rank=16 out=96 candidates=5 | exact vs CPU counter reference, max diff `0.0` | `0.0270 ms` |
| A/B | materialized B expand | rows=64 rank=64 out=1024 candidates=16 | exact scaffold path | `0.0465 ms` |
| A/B | stateless counter expand | rows=64 rank=64 out=1024 candidates=16 | kernel law guarded by CUDA parity tests | `0.0500 ms` |
| A/B | materialized B expand | rows=256 rank=64 out=1024 candidates=16 | exact scaffold path | `0.1850 ms` |
| A/B | stateless counter expand | rows=256 rank=64 out=1024 candidates=16 | kernel law guarded by CUDA parity tests | `0.0935 ms` |
| A/B | materialized B expand | rows=256 rank=128 out=4096 candidates=16 | exact scaffold path | `1.2810 ms` |
| A/B | stateless counter expand | rows=256 rank=128 out=4096 candidates=16 | kernel law guarded by CUDA parity tests | `0.4008 ms` |
| A/B | materialized B expand | rows=512 rank=128 out=4096 candidates=16 | exact scaffold path | `2.7291 ms` |
| A/B | stateless counter expand | rows=512 rank=128 out=4096 candidates=16 | kernel law guarded by CUDA parity tests | `0.7207 ms` |

Artifacts:

- `results/remote_lazy_kernel_validation/l40s_counterkernel/counter_kernel_probe/microbench_summary.json`
- `results/remote_lazy_kernel_validation/l40s_counterkernel/counter_kernel_probe/expand_ab_summary.json`
- `results/remote_lazy_kernel_validation/l40s_counterkernel/counter_kernel_probe/expand_ab_latency.png`

This is the first evidence that the final lever is plausibly the fused kernel:
the stateless expand stage removes the `B`-stack memory traffic and beats the
cached/materialized expand at larger shapes. It is still only the expand stage.
`Qx` is computed separately in the hook path, and end-to-end p128/p1024
throughput still needs a fused shrink+expand or vLLM custom-op integration.
The vLLM hook defaults to `torch_generator_field_v1` for materialized bridge
backends and to `counter_gaussian_v1` for `triton-counter`, so the old bridge
does not fall onto the slow Python SHA materialization path.

Current L40S end-to-end `triton-counter` evidence:

| run | backend | population/prompts | candidate batch | status | candidates/sec | lazy kernel s | stack s | Qx s |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| cold p16 | triton-counter, rows constexpr | p16/4 | 16 | compile-tax dominated | `0.223` | `102.861` | `0.131` | `0.209` |
| warm p16 | triton-counter, rows constexpr | p16/4 | 16 | warmed | `8.352` | `1.073` | `0.130` | `0.212` |
| warm p128 | triton-counter, rows constexpr | p128/8 | 16 | shape churn regression | `0.368` | `388.131` | `0.665` | `1.065` |
| warm p16 | triton-counter, runtime rows | p16/4 | 16 | pass | `11.112` | `0.501` | `0.123` | `0.208` |
| warm p128 | triton-counter, runtime rows | p128/8 | 16 | pass | `4.997` | `13.679` | `0.589` | `1.013` |
| same-host bridge | vLLM LoRA-kernel bridge | p128/8 | 16 | pass | `6.110` | `6.169` | `8.739` | `0.000` |
| warm p128 | triton-counter, runtime rows | p128/8 | 32 | best p128 | `12.422` | `7.661` | `0.492` | `0.847` |
| same-host bridge | vLLM LoRA-kernel bridge | p128/8 | 32 | OOM on L40S | n/a | n/a | n/a | n/a |
| warm p128 | triton-counter, runtime rows | p128/8 | 64 | pass, not faster | `12.225` | `1.556` | `0.539` | `0.855` |
| warm p1024 | triton-counter, runtime rows | p1024/8 | 32 | pass | `12.234` | `3.694` | `1.385` | `2.014` |

Artifacts:

- `results/remote_lazy_kernel_validation/l40s_counterp128/counter_p128/p128_row_runtime_cbs32/`
- `results/remote_lazy_kernel_validation/l40s_counterp128/counter_p128/p1024_row_runtime_cbs32/`
- `results/remote_lazy_kernel_validation/l40s_counterp128/counter_p128/counter_end_to_end_throughput.png`
- `results/remote_lazy_kernel_validation/l40s_counterp128/counter_p128/plots_counter_batch_parity/throughput.png`
- `results/remote_lazy_kernel_validation/l40s_counterp128/counter_p128/plots_counter_batch_parity/lazy_timing_breakdown.png`
- `results/remote_lazy_kernel_validation/l40s_counterp128/counter_p128/plots_counter_batch_parity/validation_summary.json`

The key systems bug was treating `rows` as a Triton constexpr. vLLM decode
scheduling changes row counts frequently, so this caused excessive compilation
or specialization churn at p128. `rows` must remain a runtime scalar. After
that fix, the stateless counter path is viable at larger candidate batches on
L40S because it does not carry candidate B stacks; cbs32 is the current best
measured point. The remaining bottleneck is the counter expand kernel itself:
the bridge still has a faster kernel phase at cbs16, so a production custom op
needs either a faster RNG/delta kernel or integration that removes more launch
and hook overhead.

Counter-batch replay parity is not fully closed. With p128 cbs32 as the
reference, the p1024 cbs32 prefix subset has exact replay parity across the
128 common candidates and 1024 common prompt rows. Changing candidate batch
size still causes one score mismatch out of 128 candidates:

| comparison | common candidates | score mismatches | max score diff | prompt exact match | text match |
| --- | ---: | ---: | ---: | ---: | ---: |
| p128 cbs32 vs p128 cbs16 | 128 | 1 | `0.125` | `0.9990` | `0.7295` |
| p128 cbs32 vs p128 cbs64 | 128 | 1 | `0.125` | `0.9990` | `0.8604` |
| p128 cbs32 vs p1024 cbs32 prefix | 128 | 0 | `0.0` | `1.0000` | `1.0000` |

This is acceptable for kernel-throughput exploration, but not for a strict
replay claim. The remaining correctness gate is deterministic run-level replay
under different candidate batch sizes or a documented decode determinism mode.

The Amdahl read is also more constrained than "make the expand kernel faster."
At p128 cbs32, lazy-delta time is `12.39s` inside `31.93s` scoring, so removing
the entire lazy path would cap the replay speedup at about `1.63x`; removing
only the measured kernel phase caps it at about `1.32x`. At p1024 cbs32,
lazy-delta time is `33.16s` inside `99.12s`, but measured kernel time is only
`3.69s`, so the fused/custom-op path must remove hook dispatch, separate
`Qx`, delta allocation/add, and timing/scheduling overhead too. A standalone
faster expand kernel is necessary but not sufficient.

The p128/32-prompt L40S shape also OOMed under vLLM after many candidate
chunks, despite smaller per-chunk scheduling, because the run became a KV/cache
capacity stress test. The p128 speed gate above uses 8 prompts to compare
candidate routing and lazy-delta cost without changing the subspace math.

Current L40S end-to-end `triton-counter-inplace` evidence:

| run | backend | population/prompts | candidate batch | status | candidates/sec | lazy kernel s | stack s | Qx s |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| cold p16 | triton-counter-inplace | p16/4 | 16 | compile-tax dominated | `0.801` | `28.761` | `0.165` | `0.129` |
| warm p16 | triton-counter-inplace | p16/4 | 16 | pass | `11.434` | `0.740` | `0.164` | `0.129` |
| warm p128 | triton-counter-inplace | p128/8 | 32 | pass | `13.647` | `2.851` | `0.787` | `0.578` |
| warm p128 | triton-counter-inplace | p128/8 | 64 | best p128 | `13.924` | `2.818` | `0.821` | `0.578` |
| warm p1024 | triton-counter-inplace | p1024/8 | 64 | pass | `13.808` | `5.756` | `2.744` | `1.940` |

Timing-mode correction for the same p128 cbs64 shape:

| timing mode | candidates/sec | lazy delta s | lazy kernel s | stack s | Qx s | parity |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| sync | `6.236` | `13.988` | `13.663` | `0.087` | `0.176` | reference |
| cuda-events | `16.452` | `2.686` | `2.540` | `0.089` | `0.058` | exact score and prompt parity |
| host | `16.576` | `1.108` | `0.888` | `0.081` | `0.103` | exact score and prompt parity |

The old per-phase sync timer was itself a throughput bottleneck. Future
max-throughput gates should use `OPTIMUS_LAZY_TIMING_MODE=host`; device-side
attribution should use `OPTIMUS_LAZY_TIMING_MODE=cuda-events`. Sync timing is
only a diagnostic mode.

Artifacts:

- `results/remote_lazy_kernel_validation/l40s_counter_inplace/inplace_counter/p128_cbs64/`
- `results/remote_lazy_kernel_validation/l40s_counter_inplace/inplace_counter/p1024_cbs64/`
- `results/remote_lazy_kernel_validation/l40s_counter_inplace/plots_inplace_vs_counter/throughput.png`
- `results/remote_lazy_kernel_validation/l40s_counter_inplace/plots_inplace_vs_counter/lazy_timing_breakdown.png`
- `results/remote_lazy_kernel_validation/l40s_counter_inplace/plots_inplace_batch_parity/validation_summary.md`
- `results/remote_lazy_kernel_validation/l40s_eventtiming/plots/throughput.png`
- `results/remote_lazy_kernel_validation/l40s_eventtiming/plots/lazy_timing_breakdown.png`
- `results/remote_lazy_kernel_validation/l40s_eventtiming/plots/validation_summary.md`

This is the first end-to-end positive fused-add result. Relative to the
previous best p128 out-of-place counter run, p128 cbs64 improves from
`12.225 cand/s` to `13.924 cand/s`; relative to the previous p1024 cbs32 run,
p1024 improves from `12.234 cand/s` to `13.808 cand/s`. The gain is about
`13-14%`, so it is useful but not the whole production answer.

Replay caveat: in-place cbs64 has exact score and prompt-output replay between
p128 and the p1024 prefix subset. Against the old out-of-place cbs32 reference,
the in-place path has two score mismatches out of 128 common candidates and
max score diff `0.125`. Treat it as numerically close but not a strict
replacement until the decode determinism gate is tightened. That old comparison
also predates the target-split q/v field-offset fix; the post-fix CUDA hook
suite passes `35/35`, including the regression test that local q/v field
indices write into the correct fused qkv output slices.

Post-fix synthetic L40S kernel A/B:

| rows | rank | output dim | max diff | expand+add ms | in-place add ms | speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 64 | 1024 | `0.0` | `0.0463` | `0.0304` | `1.52x` |
| 256 | 64 | 1024 | `0.0` | `0.0523` | `0.0491` | `1.06x` |
| 256 | 128 | 4096 | `0.0` | `0.3672` | `0.3763` | `0.98x` |
| 512 | 128 | 4096 | `0.0` | `0.7196` | `0.7156` | `1.01x` |

This confirms that the narrow fused-add optimization is not the whole lever:
at realistic larger rank/output shapes it is neutral on the isolated kernel.
The production path must fuse or schedule the shrink `Qx`, counter expansion,
and output addition under vLLM routing, not merely remove `main + delta`.

Current reproducible A6000 kernel-ablation evidence:

| dtype | rows | rank | output dim | total out-of-place ms | total in-place ms | total speedup | max diff | mean diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fp32 | 64 | 64 | 1024 | `0.1038` | `0.0842` | `1.232x` | `0.0` | `0.0` |
| fp32 | 256 | 64 | 1024 | `0.1348` | `0.1312` | `1.028x` | `0.0` | `0.0` |
| fp32 | 256 | 128 | 4096 | `0.7965` | `0.7806` | `1.020x` | `0.0` | `0.0` |
| fp32 | 512 | 128 | 4096 | `1.5723` | `1.5343` | `1.025x` | `0.0` | `0.0` |
| bf16 | 64 | 64 | 1024 | `0.1053` | `0.0755` | `1.395x` | `0.5` | `0.0180` |
| bf16 | 256 | 64 | 1024 | `0.1201` | `0.1178` | `1.020x` | `0.5` | `0.0179` |
| bf16 | 256 | 128 | 4096 | `0.7912` | `0.7796` | `1.015x` | `0.5` | `0.0255` |
| bf16 | 512 | 128 | 4096 | `1.5533` | `1.5240` | `1.019x` | `0.5` | `0.0253` |

Artifacts:

- `results/remote_lazy_kernel_validation/a6000_kernel_ablation/fp32/kernel_ablation_latency.png`
- `results/remote_lazy_kernel_validation/a6000_kernel_ablation/fp32/kernel_ablation_speedup.png`
- `results/remote_lazy_kernel_validation/a6000_kernel_ablation/bf16/kernel_ablation_latency.png`
- `results/remote_lazy_kernel_validation/a6000_kernel_ablation/bf16/kernel_ablation_speedup.png`

The fp32 path is exact. The bf16 max diff is add-order rounding between
PyTorch out-of-place add and Triton in-place add; mean absolute diff stays
around `0.018-0.026`.

The updated Amdahl read is that eliminating the final add/allocation helps,
but the remaining p1024 lazy path still spends `10.56s` in lazy-delta work
inside `90.57s` scoring. The next production lever remains deeper vLLM
integration: remove hook dispatch and route `Qx`, counter expand, and in-place
add as a first-class operator under vLLM scheduling.

## A6000 Run-Level Parity Post-Fix

After the in-place q/v target-split offset fix, the A6000 staged probe shows:

| probe | status | generated match | max generated logprob diff | max common top-logprob diff |
| --- | --- | ---: | ---: | ---: |
| zero scale, target-split, in-place | pass | 4/4 | `0.0000` | `0.0000` |
| nonzero, target-split, in-place | fail | 3/4 | `0.0568` | `0.7487` |
| nonzero, target-split, out-of-place counter | fail | 4/4 | `0.1201` | `0.7487` |
| nonzero, target-split, torch materialized | fail | 4/4 | `0.0776` | `0.6240` |
| nonzero, target-split, vLLM-LoRA-kernel-in-hook | fail | 4/4 | `0.0667` | `0.7487` |
| nonzero, target-split, fp32 adapter/fp32 lazy c1p1 | fail | 1/1 | `0.0467` | `0.3750` |
| nonzero, fused-qkv-exact c1p1 | fail | 1/1 | `0.0902` | `0.6250` |

Artifact root:

- `results/remote_lazy_kernel_validation/a6000_vllm_parity_postfix/`

This rules out the in-place counter kernel as the remaining strict-parity
blocker: zero-delta parity is exact, and nonzero strict adapter parity also
fails for torch materialized lazy and vLLM-LoRA-kernel-in-hook paths. The
remaining issue is hook-vs-adapter injection semantics and kernel-order drift.
Use the torch reference and injection-point checks for arithmetic development;
use vLLM adapter replay as an integration reference, not as the fused-kernel
contract itself.

## A6000 Target-Output Drift Capture

The target-output capture probe samples q/v-packed target outputs for the same
candidate and prompt under native vLLM adapter replay and lazy hook replay. It
compares the returned module outputs by candidate id, target id, and call index.

| lazy backend | generated match | max common top-logprob diff | target-output max abs | target-output mean RMS | worst layer | lazy delta s | kernel s | stack s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| triton-counter-inplace | 1/1 | `0.5011` | `1.5` | `0.0216` | 35 | `18.46` | `18.44` | `0.01` |
| vLLM-LoRA-kernel-in-hook | 1/1 | `0.4991` | `0.75` | `0.0168` | 33 | `38.24` | `0.04` | `38.11` |

Artifacts:

- `results/remote_lazy_kernel_validation/a6000_drift_capture/target_output_drift_by_layer.png`
- `results/remote_lazy_kernel_validation/a6000_drift_capture/target_output_capture_summary.json`
- `results/remote_lazy_kernel_validation/a6000_drift_capture/c1p1_targetsplit/target_output_drift.csv`
- `results/remote_lazy_kernel_validation/a6000_drift_capture/c1p1_vllm_lora_kernel/target_output_drift.csv`

Layer 0 drift is tiny (`0.00195` max abs for the in-place counter backend),
and the visible mismatch accumulates in late layers. The same strict-signature
failure class appears for both the stateless in-place counter backend and the
vLLM-LoRA-kernel-in-hook backend, so the capture does not point to a counter
kernel arithmetic bug. It points to adapter replay versus hook execution
semantics plus accumulated bf16/order drift. The speed profile still supports
the fused/custom-op lever: the counter path removes factor-stack construction,
while the bridge spends almost all measured lazy time in stack setup for this
cold c1/p1 capture.

## L40S Packed q/v Counter Kernel

The packed q/v counter kernel specializes the common fused-qkv case where the
subspace run perturbs q and v but leaves k untouched. It replaces two separate
`triton_subspace_add_counter_` launches with one launch that writes the q slice
and v slice of the fused qkv output. It supports both `target-split` and
`fused-qkv-exact` random-field policies by passing explicit q/v target hashes
and field output offsets.

Remote validation:

| check | result |
| --- | --- |
| CUDA focused suite | `40 passed` |
| fp32 packed q/v vs split q/v max diff | `0.0` for all benchmark shapes |
| bf16 packed q/v vs split q/v max diff | `0.0` for all benchmark shapes |
| p16 vLLM split vs packed `candidate_scores.jsonl` | exact after dropping timing fields |
| p16 vLLM split vs packed `per_prompt.jsonl` | exact after dropping timing fields |

Synthetic L40S q/v launch-fusion A/B:

| dtype | rows | rank | q dim | kv dim | split q/v ms | packed q/v ms | speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fp32 | 64 | 64 | 1024 | 256 | `0.0481` | `0.0306` | `1.57x` |
| fp32 | 256 | 64 | 1024 | 256 | `0.0805` | `0.0666` | `1.21x` |
| fp32 | 256 | 128 | 4096 | 1024 | `0.4941` | `0.4773` | `1.04x` |
| fp32 | 512 | 128 | 4096 | 1024 | `0.9360` | `0.9274` | `1.01x` |
| bf16 | 64 | 64 | 1024 | 256 | `0.0508` | `0.0295` | `1.72x` |
| bf16 | 256 | 64 | 1024 | 256 | `0.0802` | `0.0654` | `1.23x` |
| bf16 | 256 | 128 | 4096 | 1024 | `0.4957` | `0.4824` | `1.03x` |
| bf16 | 512 | 128 | 4096 | 1024 | `0.9462` | `0.9307` | `1.02x` |

End-to-end p16 replay on L40S:

| qkv policy | candidate replay sec | lazy delta s | lazy kernel s | stack s | Qx s |
| --- | ---: | ---: | ---: | ---: | ---: |
| split launches | `1.755` | `26.737` | `26.559` | `0.087` | `0.052` |
| packed q/v | `0.868` | `12.680` | `12.560` | `0.048` | `0.050` |

Fixed-basis p128/p1024 replay on L40S:

| population | qkv policy | candidates/sec | candidate replay sec | lazy delta s | lazy kernel s | stack s | Qx s | parity |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 128 | split launches | `4.511` | `0.222` | `22.466` | `21.892` | `0.405` | `0.171` | reference |
| 128 | packed q/v | `7.195` | `0.139` | `11.457` | `11.158` | `0.162` | `0.131` | exact |
| 1024 | split launches | `15.924` | `0.063` | `10.388` | `5.807` | `3.343` | `1.627` | reference |
| 1024 | packed q/v | `16.034` | `0.062` | `5.466` | `3.125` | `1.321` | `1.066` | exact |

The p128 and p1024 packed replays are exact against split launches for both
`candidate_scores.jsonl` and `per_prompt.jsonl` after dropping timing fields.

Artifacts:

- `results/remote_lazy_kernel_validation/l40s_qvpack/fp32/kernel_ablation_qv_speedup.png`
- `results/remote_lazy_kernel_validation/l40s_qvpack/bf16/kernel_ablation_qv_speedup.png`
- `results/remote_lazy_kernel_validation/l40s_qvpack/vllm_p16_split-launches/summary.json`
- `results/remote_lazy_kernel_validation/l40s_qvpack/vllm_p16_packed-qkv/summary.json`
- `results/remote_lazy_kernel_validation/l40s_qvpack_p128/plots_packed_replay/throughput.png`
- `results/remote_lazy_kernel_validation/l40s_qvpack_p128/plots_packed_replay/lazy_timing_breakdown.png`

This is a real kernel-aligned improvement and it removes one obvious launch
overhead in the q/v target preset. It is not the final Amdahl lever. At p1024,
packed q/v halves lazy-delta time but improves end-to-end fixed-basis replay
only from `15.924` to `16.034 cand/s`, because model rollout and scheduling now
dominate. The production path still needs a vLLM custom-op/scheduling
integration that reduces work around the kernel, not only a faster q/v add.

Current A6000 direct `Qx + counter add` evidence:

The guarded `triton_subspace_add_counter_qv_from_x_` prototype directly takes
activation rows `x`, basis `Q`, candidate seeds, q/v target hashes, and the
fused qkv output buffer. It computes local `Qx` inside the counter-add kernel
and writes q/v while preserving the K slice. Runtime access is intentionally
behind `OPTIMUS_LAZY_QKV_KERNEL_POLICY=packed-qkv-from-x`; it is not the
default path.

Remote validation:

| check | result |
| --- | --- |
| CUDA focused suite | `42 passed` |
| fp32 direct fused-from-`x` max diff | `3.8e-05` to `9.9e-05` |
| bf16 direct fused-from-`x` max diff | `0.5` to `1.0`, consistent with low-precision accumulation/add-order differences |
| fetched PNG headers | valid |

Synthetic A6000 direct-fusion A/B:

| dtype | rows | rank | q dim | kv dim | `Qx + packed q/v` ms | fused from `x` ms | speed vs packed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fp32 | 64 | 64 | 1024 | 256 | `0.122` | `1.884` | `0.065x` |
| fp32 | 256 | 64 | 1024 | 256 | `0.152` | `5.810` | `0.026x` |
| fp32 | 256 | 128 | 4096 | 1024 | `1.004` | `41.398` | `0.024x` |
| fp32 | 512 | 128 | 4096 | 1024 | `1.958` | `91.148` | `0.021x` |
| bf16 | 64 | 64 | 1024 | 256 | `0.115` | `1.854` | `0.062x` |
| bf16 | 256 | 64 | 1024 | 256 | `0.139` | `4.544` | `0.031x` |
| bf16 | 256 | 128 | 4096 | 1024 | `0.990` | `32.478` | `0.030x` |
| bf16 | 512 | 128 | 4096 | 1024 | `1.919` | `62.311` | `0.031x` |

Artifacts:

- `results/remote_lazy_kernel_validation/a6000_qxcounter/fp32/kernel_ablation_qv_fused_from_x.png`
- `results/remote_lazy_kernel_validation/a6000_qxcounter/bf16/kernel_ablation_qv_fused_from_x.png`
- `results/remote_lazy_kernel_validation/a6000_qxcounter/fp32/kernel_ablation_summary.json`
- `results/remote_lazy_kernel_validation/a6000_qxcounter/bf16/kernel_ablation_summary.json`

This rules out the naive output-tiled single-kernel implementation. It
recomputes `Qx` for every output tile, which overwhelms the benefit of avoiding
the intermediate `z` tensor. A viable production `Qx + counter add` path needs
one `Qx` computation per activation-site row block followed by scheduled
counter add, likely as a vLLM custom-op/two-stage shrink-expand pipeline with
explicit row-candidate routing.

Current A6000 shared-`Qx` cache evidence:

The hook runtime now has an identity-guarded `Qx` cache for sibling target
modules that read the exact same activation tensor and activation site. Local
tests prove reuse for separate q/v modules and prove that different tensor
objects do not alias through the cache.

Remote validation:

| check | result |
| --- | --- |
| local focused hook/kernel suite | `35 passed`, `9` CUDA skips on Mac |
| remote A6000 focused hook/kernel suite | `44 passed` |
| p128 cache-on warm qv replay | `9.400 cand/s`, `qx_cache_hits=0`, `qx_cache_misses=4752` |
| p128 cache-off qv replay | `9.383 cand/s`, `qx_cache_hits=0`, `qx_cache_misses=4752` |
| p128 cache-on warm `Qx` time | `0.465s` inside `13.617s` candidate replay |
| p128 cache-on warm lazy-kernel time | `2.140s` |

Artifacts:

- `results/remote_lazy_kernel_validation/a6000_qxcache/p128_cache_on/summary.json`
- `results/remote_lazy_kernel_validation/a6000_qxcache/p128_cache_off/summary.json`
- `results/remote_lazy_kernel_validation/a6000_qxcache/p128_cache_on_warm/summary.json`

This is a negative for Qwen/vLLM qv throughput, not a correctness failure.
Qwen's vLLM path exposes q/k/v as one fused `qkv_proj`, and the packed q/v
counter backend already computes `Qx` once per fused-qkv hook. Therefore the
sibling-cache lever has no hits on the main p128 qv path. The remaining useful
work is still `Qx + counter add`, but it must be a first-class vLLM
row-block/custom-op scheduling path that removes Python hook dispatch and
coordinates one activation-site projection with packed counter add.

Current A6000 row-mapping cache evidence:

The stateless counter backends now reuse the same row-candidate mapping cache
used by the vLLM LoRA-kernel bridge. This avoids repeatedly copying the same
stable `row_candidate_id` tensor to device for decode shapes that recur under
vLLM scheduling. `OPTIMUS_LAZY_ROW_MAPPING_CACHE_SIZE=0` disables the cache for
A/B testing.

Remote validation:

| check | result |
| --- | --- |
| local focused hook/kernel suite | `43 passed`, `9` CUDA skips on Mac |
| remote A6000 focused hook/kernel suite | `52 passed` |
| p128 cache-on row-map hits/misses | `4760/136` |
| p128 cache-off row-map hits/misses | `0/4896` |

p128 replay on A6000, cbs64, 8 prompts, rank64 q/v, packed q/v counter add:

| row-map cache | candidates/sec | sec/candidate | lazy delta s | kernel s | stack s | Qx s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| on | `7.648` | `0.1308` | `5.990` | `0.607` | `0.163` | `0.398` |
| off | `7.369` | `0.1357` | `6.188` | `0.628` | `0.321` | `0.431` |

Artifacts:

- `results/remote_lazy_kernel_validation/a6000_rowmap/p128_cbs64_cache_on/summary.json`
- `results/remote_lazy_kernel_validation/a6000_rowmap/p128_cbs64_cache_off/summary.json`

This is worth keeping but it is not the main lever. It trims metadata movement
and stack time, while the broader throughput gap remains in Python hook
dispatch, vLLM scheduling, and the absence of a first-class row-block custom
operator for `Qx + counter add`.

## Benchmark Ladder

All benchmark rows must record hardware, vLLM version, FlashInfer version,
model, prompt count, output tokens, rank, target preset, candidate batch size,
and exact command.

1. `p16` bridge smoke, rank 64, candidate batch 16.
2. `p128` bridge speed gate, rank 64, candidate batch 16.
3. Native vLLM subspace-as-LoRA `p128` baseline, rank 64, max_loras 16.
4. Fused-kernel microbench on captured vLLM row shapes.
5. Fused-kernel `p128` end-to-end replay.
6. `p1024` only after the fused p128 result beats or materially matches the
   bridge and native adapter baselines.

Current A100 p128 bridge evidence:

| run | candidates/sec | output tok/sec | best | base |
| --- | ---: | ---: | ---: | ---: |
| no-adapter repeat p16 | 1.486 | 3597 | 11/128 | 11/128 |
| native subspace-as-LoRA p128 r64 | 1.162 | 2710 | 25/128 | 11/128 |
| true-lazy vLLM-kernel bridge p128 r64 | 1.310 | 3046 | 25/128 | 11/128 |
| old chunked lazy p128 effective-r16 | 0.071 | 169 | 27/128 | 12/128 |

Bridge p128 timing:

| phase | seconds |
| --- | ---: |
| total replay | 97.71 |
| lazy delta | 49.34 |
| vLLM LoRA kernels | 20.56 |
| metadata | 8.22 |
| factor stack | 10.14 |

## Fused Kernel Hypothesis

The fused kernel should replace:

```text
z = Q x
delta = G z
y += beta * delta
```

as a single row-candidate-routed operation, avoiding separate shrink/expand
launches and avoiding materialized full adapters. The first fused PR should
target the measured bridge bottleneck rather than broad vLLM executor changes:

- stable row-candidate routing;
- candidate-major blocks for the common packed layout;
- fallback correctness path for arbitrary row order;
- deterministic random-field replay or cached factor tiles with stable hashes;
- fp16/bf16 input/output with fp32 accumulation where measurable;
- parity against the torch reference and current vLLM-kernel bridge.

Do not optimize the Python hook further unless profiling shows it dominates
after the fused delta path lands.
