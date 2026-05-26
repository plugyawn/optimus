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
| nonzero scale | torch | 2.0 | fail | 2/2 | 0.0183 | 0.1532 |
| nonzero scale | torch fp32 compute | 2.0 | fail | 2/2 | 0.0805 | 0.2499 |
| nonzero scale | vLLM LoRA-kernel bridge | 2.0 | fail | 2/2 | 0.0796 | 0.2498 |

The zero-scale pass shows the probe is not dominated by run-to-run vLLM noise.
The nonzero failures mean strict adapter-vs-lazy logprob parity is still open.
Do not convert this bridge into a production claim; use it to localize the
exact semantic gap that the fused/custom-op path must close.

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
