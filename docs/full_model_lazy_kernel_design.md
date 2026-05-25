# Activation-Site Projected RandOpt Lazy Kernel

## Status And Scope

This document is the implementation source of truth for Optimus
transformer-linear subspace RandOpt over vLLM. The public method name is
`subspace`. The precise mechanism is activation-site projected RandOpt: random
perturbations are applied to transformer block linear modules through activation
bases captured at the sites those modules read from.

The search-time operation for target module `t` and candidate `c` is:

```text
y_t,c(x) = W_t x + delta_y_t,c(x)
delta_y_t,c(x) = beta_t G_t,c Q_s(t) x
```

Where:

- `s` is an activation site.
- `t` is a target linear module that reads from activation site `s(t)`.
- `Q_s` is an orthonormal activation basis for site `s`.
- `G_t,c` is a deterministic candidate-specific standard Gaussian random field.
- `beta_t` is the resolved scale for target `t`.
- `c` is the candidate identity.

Algebraically, each candidate induces a low-rank perturbation
`Delta W_t,c = beta_t G_t,c Q_s(t)`, but search-time execution is lazy:
Optimus injects `beta_t G_t,c Q_s(t) x` inside vLLM model execution and never
loads, swaps, or materializes per-candidate adapters in the hot path.

The primary serving artifact is a lazy top-K candidate ensemble. Single-winner
materialized exports remain useful for replay, debugging, distillation, and
baselines, but they are not the primary search-time representation.

## Goals And Non-Goals

Goals:

- Search activation-site projected RandOpt candidates over transformer linear
  targets.
- Support target presets `qv`, `attn-qkvo`, `mlp`, and
  `transformer-linears`.
- Keep vLLM as the production execution substrate. vLLM owns model loading,
  scheduling, batching, attention, and KV-cache storage. Optimus owns candidate
  identity, activation bases, lazy perturbation semantics, scoring, and run
  artifacts.
- Avoid PEFT/vLLM adapter materialization, adapter loading, and adapter
  swapping in the search hot path.
- Treat top-K lazy ensemble serving as a first-class runtime path.
- Preserve exact replayability from candidate seed, sign, basis hash, scale
  mode, radius, budget policy, target set, and runtime config.
- Keep v1 single-GPU-per-worker while defining candidate-shard boundaries for a
  later one-worker-per-GPU PR.
- Make the cheap scientific gating experiment possible before optimized kernels
  are implemented.

Non-goals for v1:

- Tensor-parallel lazy kernels.
- Replacing vLLM, SGLang, LightEval, PyTorch, Triton, FlashInfer, or standard
  evaluation harnesses.
- Claiming exact dense RandOpt outside the captured activation subspace.
- Serving arbitrary structured-generation ensembles with a universal
  majority-vote rule.
- Two-sided output bases. They may be useful later, but are not needed for v1.
- Quantized-base parity claims. v1 parity claims require unquantized fp16/bf16
  base weights unless a later quantization policy proves equivalence.
- Making LoRA terminology part of the subspace public API.

## Activation Sites And Targets

A decoder block has several activation read sites. The default v1 policy is:

| activation site | target modules using it | basis |
| --- | --- | --- |
| `layer_i.attn_in` | `q_proj`, `k_proj`, `v_proj` | shared `Q_i,attn` |
| `layer_i.o_in` | `o_proj` | `Q_i,o` |
| `layer_i.mlp_in` | `gate_proj`, `up_proj` | shared `Q_i,mlp` |
| `layer_i.down_in` | `down_proj` | `Q_i,down` |

A standard decoder block therefore has roughly four activation bases and seven
target random fields per layer. A target module must not own a duplicate basis
when it reads from an existing activation site unless the run explicitly opts
into a non-default basis family.

Every activation site must be represented by an `ActivationSiteSpec` before it
can be used by a backend. Required fields are:

- `site_id`;
- architecture family, for example `qwen3_text` or `llama_decoder`;
- layer index and owning block path;
- exact read tensor path and hook point;
- pre/post RMSNorm or LayerNorm location;
- tensor shape convention `[tokens, hidden]` after any sequence flattening;
- runtime dtype and accumulation dtype;
- tensor-parallel sharding policy;
- target modules that read the site;
- calibration prompt ids hash and decode/config hash.

The v1 Qwen/Llama mapping is: `attn_in` is the normalized attention input read
by q/k/v projections, `o_in` is the attention output-projection input after
attention and before `o_proj`, `mlp_in` is the normalized MLP input read by
gate/up projections, and `down_in` is the post-activation gated MLP tensor read
by `down_proj`. Any architecture that does not expose these tensors with the
same semantics needs an explicit mapping entry and tests before it can use the
`transformer-linears` preset.

The default full transformer preset is `transformer-linears`. It means selected
dense transformer block linear maps. It excludes embeddings, LM head, layer
norms, routers, and architecture-specific nonstandard modules unless explicitly
added through `--targets`.

## Projection Contract

Let `Q_s` have orthonormal rows and define `P_s = Q_s.T Q_s`. For one target
module, let a dense Gaussian perturbation have entries sampled from
`N(0, sigma_w^2)`. Let the subspace perturbation be `Delta W_Q = sigma_w G Q_s`
with standard Gaussian `G`. Then, for every fixed input `x`:

```text
Delta W_Q x  has the same distribution as  Delta W_dense P_s x
```

The lazy perturbation is dense Gaussian RandOpt evaluated on projected input
`P_s x`. It discards exactly the component of the input outside the span of
`Q_s`, before downstream nonlinear drift.

For activation matrix `X_s` with rows as calibration token activations
(`[tokens, d_s]`), dense output-effect covariance across tokens for one output
coordinate is:

```text
sigma_w^2 X_s X_s.T
```

Projected output-effect covariance is:

```text
sigma_w^2 X_s Q_s.T Q_s X_s.T
```

Dense equivalence holds only on the captured activation subspace. If downstream
perturbations move later activations outside the basis, that is additional
approximation error and must be reported by drift diagnostics.

## Basis Construction

Supported v1 basis kinds:

```text
activation-svd
random-orthonormal
shuffled-activation-svd
```

`activation-svd` is the intended method. `random-orthonormal` and
`shuffled-activation-svd` are mandatory controls. Future residual-biased,
gradient-biased, or task-biased bases are allowed only as separately reported
families because they weaken the direct projected-dense claim.

Control definitions are fixed:

- `activation-svd`: top right singular vectors of the uncentered or explicitly
  centered calibration activation matrix, as selected by `--basis-centering`.
- `random-orthonormal`: draw a deterministic Gaussian matrix from
  `basis_control_seed`, QR-orthonormalize rows in fp64, then store rows in the
  runtime basis dtype.
- `shuffled-activation-svd`: independently permute each activation feature
  column across calibration tokens with `basis_control_seed`, then run the same
  SVD pipeline. This preserves per-coordinate marginal scale while destroying
  token-level feature covariance.

Each basis artifact records the control seed, prompt ids hash, scorer/task
config hash, decode config hash, calibration token count, and whether the run is
transductive or non-transductive.

Basis centering is explicit:

```text
--basis-centering none | mean
```

Default v1 is `none`, because the projected-dense covariance claim is about raw
activations, not centered activations. Centered SVD is a valid ablation and must
be reported separately.

Basis token source is explicit:

```text
--basis-token-source prefill | decode | prefill+decode
```

Default v1 is `prefill`. For generation-heavy tasks, `prefill+decode` should be
used when feasible. Runs that use decode evaluation must report captured energy
separately on prefill and decode probes.

The basis artifact must report:

```text
basis_split = train | screen_unlabeled | public_unlabeled | holdout_forbidden
```

Holdout labels must never be used for basis construction. If unlabeled holdout
prompts are used, the run must mark this as transductive and must not compare
against non-transductive baselines without disclosure.

## Captured Energy And Output Power

For calibration activations `x ~ D_s`, define:

```text
H_s(r) = E ||Q_s x||^2
A_s    = E ||x||^2
gamma_s(r) = H_s(r) / A_s
```

If `Q_s` is formed from the top right singular vectors of the uncentered
activation matrix, `H_s(r)` is the mean sum of the top-r squared singular values
over calibration tokens.

For target `t`, define base output power:

```text
P_t = E ||W_t x||^2, where x ~ D_s(t)
```

The basis collection pass must accumulate `P_t` for every target. This is cheap
and required for relative-output-RMS scale mode.

## Scale Modes And Budget Policy

The old `--sigma` interface is ambiguous and is not part of the final subspace
API. v1 supports exactly two scale modes.

### Projected Dense

This mode is for dense-approximation claims:

```bash
--scale-mode projected-dense
--sigma-w-grid 1e-5,3e-5,1e-4
```

The resolved target scale is `beta_t = sigma_w tau_t`. With
`--budget-policy raw-dense`, `tau_t = 1`, so the method samples the dense
Gaussian law restricted to each target's activation subspace. Increasing rank
increases perturbation RMS according to captured energy:

```text
E ||delta_y_t||^2 = sigma_w^2 m_t H_s(t)(r)
```

This rank/RMS coupling is the honest projected-dense law. If any normalized
budget policy is used with `projected-dense`, the run must be described as
budgeted projected dense, not isotropic dense RandOpt.

### Relative Output RMS

This is the default search mode and the right mode for fair rank and
target-preset comparisons:

```bash
--scale-mode relative-output-rms
--rho-grid 0.002,0.005,0.01,0.02
```

Let `m_t` be target output dimension and let budget weights `w_t` satisfy
`sum_t w_t^2 = 1`. The resolved target scale is:

```text
beta_t = rho w_t sqrt(P_t / (m_t H_s(t)))
```

Then:

```text
E_x,G ||delta_y_t(x)||^2 = rho^2 w_t^2 P_t
```

Thus `rho w_t` is the target's relative output RMS perturbation, and `rho` is
the global perturbation radius under the selected budget policy. This fixes the
confound where larger rank, larger captured energy, or a larger target set
silently changes perturbation magnitude.

Degenerate calibration stats must fail closed unless explicitly overridden:
zero effective rank, tiny `H_s`, zero/NaN `P_t`, or unstable scale estimates
must block `relative-output-rms`.

### Budget Policies

Supported v1 policies:

```text
raw-dense
per-target-equal
per-layer-equal
per-block-equal
custom-json
```

Semantics:

- `raw-dense`: valid default only for `projected-dense`; does not normalize
  total target-set energy.
- `per-target-equal`: `w_t = 1 / sqrt(|T|)`.
- `per-layer-equal`: each selected layer receives equal squared budget; targets
  within a layer split that layer's budget equally.
- `per-block-equal`: attention and MLP blocks receive equal squared budget per
  layer; targets inside each block split equally.
- `custom-json`: user-provided nonnegative weights, normalized by Optimus so
  `sum_t w_t^2 = 1`.

Default v1 search setting:

```bash
--scale-mode relative-output-rms
--budget-policy per-block-equal
```

`per-target-equal` remains available as a simple ablation. Early-layer
perturbations propagate through more computation than late-layer perturbations,
so end-to-end logit-KL and drift diagnostics are required for positive results
under any budget policy.

## Candidate Random Fields

The candidate random field is a pure deterministic function:

```text
G_abs_t,c[j,a] =
  GaussianHash(rng_version, direction_seed, target_id, j, a, salt)
G_t,c[j,a] = sign * G_abs_t,c[j,a]
```

`G` is not a persistent artifact. Runtime implementations may cache
materialized tiles or candidate-block tensors in HBM when useful, but replay
semantics must come from candidate identity and indices, not from stored tensors
or mutable RNG state. Antithetic candidates share `G_abs` exactly and differ
only by the outer sign.

`gaussian_hash_v1` is specified as follows. Canonicalize `target_id` as the
fully qualified post-alias module path from `TargetModule.target_id`, using `/`
only as a separator inside model ids and `.` inside module paths, with no
leading/trailing whitespace. Encode `rng_version`, `direction_seed`, canonical
`target_id`, row index `j`, basis index `a`, and `salt` as UTF-8 fields separated
by NUL bytes in exactly that order. Hash the payload with SHA-256, interpret the
first two uint64 little-endian words as open-interval uniforms by adding `0.5`
and dividing by `2**64`, and use a Box-Muller transform to produce one fp64
standard normal. Apply sign in fp64, then cast to the requested runtime dtype at
the kernel boundary. Implementations may replace this with a counter-based RNG
only after golden-vector tests prove bitwise equality for the published
`gaussian_hash_v1` values or after the `rng_version` is changed.

Allowed implementations:

- v1 torch reference may materialize small tiles or per-target candidate blocks.
- Production kernels must be able to generate deterministic Gaussian tiles from
  a counter-based RNG or equivalent stateless random field.
- Batch size, candidate block composition, process count, and scheduler order
  must not change any candidate's random-field values.
- Golden-vector tests must pin representative values for multiple target ids,
  candidate ids, signs, and indices on CPU and GPU.

Forbidden behavior:

- resampling `G` per token;
- generating `G` from mutable global RNG state;
- making `G` depend on candidate batch index instead of stable candidate id;
- caching `G` in a way that changes replay after process restart.

Candidate identity contains:

```json
{
  "candidate_id": "stable string or integer",
  "direction_seed": 12345,
  "sign": "+",
  "basis_hash": "...",
  "target_set_hash": "...",
  "scale_mode": "relative_output_rms",
  "rho_or_sigma_w": 0.01,
  "budget_policy": "per_block_equal",
  "budget_hash": "...",
  "rng_version": "gaussian_hash_v1",
  "runtime_dtype": "bf16"
}
```

For antithetic sampling, candidates with the same `direction_seed` and opposite
signs share the same absolute random field and differ only by global sign.

## Search And Top-K Ensembles

For each radius, rank, basis kind, and target preset, search evaluates a
population of candidates on a screen split.

Mandatory outputs:

- base score on screen;
- candidate screen scores;
- candidate holdout scores for selected candidates only;
- best single-candidate screen and holdout scores;
- top-K ensemble screen and holdout scores;
- diversity metrics for selected K;
- bootstrap confidence intervals when the scorer permits.

Holdout is for confirmation, not candidate selection. If K or radius is selected
using holdout, this must be reported as validation-set tuning and cannot be
treated as final test performance.

The lazy top-K ensemble artifact is first-class:

```text
top_k_ensemble.json
```

It contains:

```json
{
  "ensemble_kind": "lazy_top_k",
  "schema_version": "top_k_ensemble_v1",
  "aggregation": "majority-vote | mean-logprob | score-sum | custom",
  "tie_break_policy": "lowest_candidate_id",
  "selection_rule": "screen_top_k_fixed_config",
  "K": 16,
  "candidates": [
    {
      "candidate_id": "seed12345:+:rho0.01",
      "direction_seed": 12345,
      "sign": "+",
      "basis_hash": "...",
      "target_set_hash": "...",
      "scale_mode": "relative_output_rms",
      "rho_or_sigma_w": 0.01,
      "budget_policy": "per_block_equal",
      "budget_hash": "...",
      "rng_version": "gaussian_hash_v1",
      "runtime_dtype": "bf16"
    }
  ],
  "basis_hash": "...",
  "scale_mode": "relative_output_rms",
  "rho_or_sigma_w": 0.01,
  "target_set_hash": "...",
  "scorer_version": "...",
  "prompt_ids_hash": "...",
  "basis_collection_config_hash": "...",
  "runtime_config_hash": "...",
  "decode_config_hash": "..."
}
```

The same lazy kernel used to score candidate blocks must serve top-K ensembles.
For fixed-choice/logprob scoring and identical prefill, serving K candidates
over one prompt has the same computational shape as search over a K-candidate
block. For open-ended autoregressive generation, candidate continuations can
diverge; the runtime must keep candidate identity on every generated row and
cannot assume shared decode trajectories beyond prefixes that are actually
identical under the candidate-keyed cache policy.

Supported v1 aggregation modes:

```text
majority-vote
mean-logprob
score-sum
```

`majority-vote` is valid only when candidate outputs are discrete comparable
answers. For open-ended structured generation, v1 must use task-specific
aggregation, logprob aggregation over fixed choices, reranking, or distillation.

Report for top-K:

- pairwise answer disagreement;
- pairwise logprob/logit correlation on fixed probes;
- pairwise Jensen-Shannon divergence or symmetric KL where available;
- number of distinct answers among K;
- oracle-top-K score if an oracle answer is available;
- marginal ensemble gain over best single candidate.

Default v1 selector is top-K by screen score plus post-hoc diversity reporting.
A diversity-aware selector is allowed only as an explicitly reported ablation.

## Required Scientific Gate

Before implementing Triton kernels, two-sided bases, or elaborate quantization
paths, Optimus must run the subspace-density gate:

```text
basis_kind in {activation-svd, random-orthonormal, shuffled-activation-svd}
scale_mode = relative-output-rms
budget_policy = per-block-equal or a predeclared alternative
basis_rank in predeclared grid
rho in predeclared grid
population = fixed
seed_panel = fixed
screen_split = fixed
holdout_split = fixed
K_grid = fixed
```

Question: does activation-SVD concentrate useful directions better than a random
orthonormal basis of the same rank at fixed radius, fixed target budget, fixed
population, fixed scorer, fixed prompts, and fixed seeds?

Report:

- improvement probability `Pr[S(c) > S(base)]`;
- best single-candidate score;
- top-K ensemble score;
- screen-to-holdout drop;
- top-K diversity metrics;
- antithetic odd/even diagnostics;
- captured activation energy;
- optional gradient-capture ratio.

When a differentiable surrogate objective `J` exists, report gradient capture:

```text
kappa_t(Q) = ||grad_W_t J Q_s(t).T||_F^2 / ||grad_W_t J||_F^2
```

Activation energy capture measures approximation fidelity. Gradient capture
measures objective relevance. They are not the same.

The primary scientific gate is top-K ensemble holdout score at one locked
configuration: one K, one rank, one radius, one target preset, one task panel,
one seed panel, one scorer, and one aggregation rule. That configuration must be
declared before production-gate holdout scores are read. If a rank/radius/K grid
is explored, it is a calibration or validation search; either choose the locked
configuration on a separate validation split before the gate, or apply the same
screen-only selection rule to every basis family and use a predeclared
multiple-comparison correction such as Holm-Bonferroni over all tested
configuration contrasts.

Proceed to optimized kernels for a scientific win only if activation-SVD has a
paired bootstrap 95% confidence interval lower bound above zero versus both
random-orthonormal and shuffled-SVD controls on the locked primary metric. A
statistically indistinguishable tie may authorize an engineering proceed, but it
must be labeled `engineering_proceed_no_scientific_win` and requires one
predeclared operational advantage at equal quality: at least 25% lower
logit-KL/drift, at least 20% lower lazy overhead, or at least 10 percentage
points higher captured activation energy. Secondary metrics are reported for
diagnosis only: improvement density, best single-candidate holdout score,
diversity-adjusted ensemble gain, and screen-to-holdout drop.

If holdout is used to choose rank, radius, K, target preset, or basis family,
that split becomes validation, not final test evidence. Final claims then need a
fresh locked test split or an external benchmark lane. If activation-SVD is not
better than random `Q` under this policy, optimized kernels are not the next
bottleneck; the basis family is.

## vLLM Runtime Integration

Optimus adds a lazy perturbation layer around vLLM execution. It must not fork
or reimplement vLLM's executor, tokenizer, scheduler, attention kernels, paged
KV-cache handling, or batching.

V1 supports two implementation lanes:

```text
eager-wrapper
custom-op
```

`eager-wrapper` is acceptable for initial parity and gating experiments.
`custom-op` or equivalent registered operator integration is required before
relying on CUDA-graph-compatible production performance.

The backend contract is fail-closed. V1 pins a tested vLLM version range in the
serving extra, records the exact vLLM version in every run artifact, and refuses
to enable subspace search when the expected model-runner hooks or request
metadata path are unavailable. The intended request metadata path is vLLM custom
arguments: offline `SamplingParams.extra_args` and online `vllm_xargs`, which
vLLM maps to `SamplingParams.extra_args`. Optimus stores candidate identity in
that metadata, then expands it to explicit `row_candidate_id` tensors at the
model-runner/linear-op boundary. The intended optimized operator path is vLLM's
registered `CustomOp` mechanism or an equivalent out-of-tree custom op, not a
fork of the executor or scheduler. Optimized fast paths may use candidate-major
row layout only after a routing parity test proves equivalence to explicit row
metadata. CUDA graph capture is disabled for the eager wrapper; custom-op graph
capture is a separate acceptance gate.

For each target module:

1. receive activation rows `x_n`;
2. receive explicit `row_candidate_id[n]` metadata;
3. compute or reuse `z_n = Q_s(t) x_n`;
4. generate/apply `G_t,c(n)` for that row's candidate id;
5. add `beta_t G_t,c(n) z_n` to the base linear output.

The implementation may use candidate-major fast paths only after verifying row
layout. The correctness path must use explicit row-to-candidate metadata.

Every activation row entering a perturbed target must carry candidate identity.
Candidate identity must never be inferred solely from physical row order. This
is required because continuous batching, ragged prompts, prefill/decode mixing,
scheduler behavior, and graph capture can change row layout without changing
model semantics.

Any candidate-specific perturbation that can affect hidden states contributing
to cached K/V or prefix state invalidates cache sharing, including q/k/v/o/MLP
targets when they occur before or inside the cached computation. Therefore:

- candidate-different requests must not share KV-cache entries;
- candidate identity must be part of prefix-cache validity; or
- prefix caching must be disabled during search and ensemble serving.

Default v1:

```bash
--prefix-cache-policy disabled-for-search
```

A future `candidate-keyed` policy is allowed only if the cache key includes all
candidate fields that affect computation: base model hash, basis hash,
candidate id, target set hash, scale mode, radius, budget hash, RNG version, and
runtime dtype.

V1 exact parity claims require unquantized fp16/bf16 base weights. Quantized
base search may be allowed experimentally, but must report base weight dtype,
runtime dtype, lazy delta dtype, quantization kind, and materialization policy.
A bf16 lazy delta added to a quantized base linear is not automatically
equivalent to materializing and quantizing `W + Delta W`.

## Public CLI

Canonical search command:

```bash
optimus search \
  --backend vllm \
  --method subspace \
  --model MODEL \
  --data DATA \
  --out OUT \
  --population 1024 \
  --basis-rank 128 \
  --target-preset transformer-linears \
  --layers all \
  --scale-mode relative-output-rms \
  --rho-grid 0.002,0.005,0.01,0.02 \
  --budget-policy per-block-equal \
  --basis-kind activation-svd \
  --basis-centering none \
  --basis-token-source prefill \
  --top-k-grid 1,4,8,16 \
  --antithetic \
  --candidate-batch-size auto \
  --kernel torch
```

Projected-dense science command:

```bash
optimus search \
  --backend vllm \
  --method subspace \
  --model MODEL \
  --data DATA \
  --out OUT \
  --population 1024 \
  --basis-rank 128 \
  --target-preset transformer-linears \
  --layers all \
  --scale-mode projected-dense \
  --sigma-w-grid 1e-5,3e-5,1e-4 \
  --budget-policy raw-dense \
  --basis-kind activation-svd \
  --top-k-grid 1,4,8,16 \
  --antithetic \
  --candidate-batch-size auto \
  --kernel torch
```

Deprecated terms to remove:

```text
--rank          use --basis-rank
--sigma         use --sigma-w only with projected-dense
--sigma-grid    use --sigma-w-grid or --rho-grid
full-model      use transformer-linears
```

Forbidden in the subspace public API:

```text
max_loras
chunk_adapters
adapter_dir
method=lora for subspace execution
LoRARequest in the search hot path
```

## Artifact Contract

A run writes:

| file | purpose |
| --- | --- |
| `subspace_state.pt` | Versioned activation-site basis tensors and calibration stats. |
| `subspace_state_summary.json` | Human-readable basis and scale metadata. |
| `candidates.jsonl` | Candidate identities before or during scoring. |
| `candidate_scores.jsonl` | Per-candidate screen and selected holdout scores. |
| `top_k_ensemble.json` | Primary top-K lazy ensemble artifact. |
| `summary.json` | Run-level summary. |
| `validation_report.json` | Gating, controls, bootstrap, diversity, and drift reports. |
| `systems_report.json` | Throughput and memory report. |
| `exports/` | Optional single-winner or distillation artifacts only. |

All JSON artifacts include `schema_version`, `created_at`, `optimus_version`,
`git_commit`, `git_dirty`, `command`, `environment`, `model_id_or_path`,
`model_revision`, `tokenizer_hash`, `task_config_hash`, `decode_config_hash`,
`prompt_contract_hash`, `screen_split_hash`, and `holdout_split_hash` unless the
artifact is explicitly marked as a child artifact that references `summary.json`
by hash.

`subspace_state.pt` required fields:

```json
{
  "schema_version": "subspace_state_v1",
  "model_id_or_path": "...",
  "model_revision": "...",
  "tokenizer_hash": "...",
  "prompt_contract_hash": "...",
  "target_preset": "transformer-linears",
  "explicit_targets": [],
  "layers": "all",
  "basis_kind": "activation-svd",
  "basis_centering": "none",
  "basis_token_source": "prefill",
  "basis_split": "train",
  "activation_sites": [
    {
      "site_id": "layer_17.attn_in",
      "input_dim": 4096,
      "requested_rank": 128,
      "effective_rank": 128,
      "basis_tensor_key": "...",
      "singular_values": "...",
      "captured_energy": 0.91,
      "prefill_captured_energy": 0.91,
      "decode_captured_energy": null,
      "H_s": 1234.5,
      "A_s": 1357.9,
      "orthonormality_error": 1e-5,
      "gram_error": 1e-5,
      "num_calibration_tokens": 200000
    }
  ],
  "targets": [
    {
      "target_id": "layer_17.self_attn.q_proj",
      "activation_site_id": "layer_17.attn_in",
      "output_dim": 4096,
      "base_output_power_P_t": 5678.9
    }
  ]
}
```

`summary.json` required scale and budget fields:

```json
{
  "kind": "subspace_search",
  "backend": "vllm",
  "method": "subspace",
  "scale_mode": "relative_output_rms",
  "rho_grid": [0.002, 0.005, 0.01, 0.02],
  "sigma_w_grid": null,
  "budget_policy": "per_block_equal",
  "resolved_target_scales": [
    {
      "target_id": "layer_17.self_attn.q_proj",
      "budget_weight": 0.0714,
      "beta_t_by_radius": {"0.01": 0.000083}
    }
  ],
  "rng_version": "gaussian_hash_v1",
  "candidate_routing": "row_candidate_id",
  "prefix_cache_policy": "disabled_for_search",
  "kernel": "torch"
}
```

`candidates.jsonl` rows contain the complete `SubspaceCandidate` identity:
`candidate_id`, `direction_seed`, `sign`, `basis_hash`, `target_set_hash`,
`scale_mode`, `rho_or_sigma_w`, `budget_policy`, `budget_hash`, `rng_version`,
`runtime_dtype`, `radius_index`, `target_preset`, and `basis_rank`.

`candidate_scores.jsonl` rows contain `candidate_id`, split, scorer name and
version, aggregate metrics, sample count, prompt ids hash, decode config hash,
elapsed time, token counts, and optional path to per-sample rows. Selected
holdout rows must include the selector that promoted the candidate.

`top_k_ensemble.json` contains the full candidate identities, not only ids, plus
`aggregation`, `tie_break_policy`, `selection_rule`, `K`, `scorer_version`,
`prompt_ids_hash`, `basis_collection_config_hash`, `runtime_config_hash`,
`decode_config_hash`, and replay hashes for `subspace_state.pt` and
`candidate_scores.jsonl`.

`validation_report.json` has separate sections for math tests, RNG/replay tests,
routing/cache tests, selector quality, holdout quality, ensemble quality, drift
diagnostics, random/shuffled controls, and throughput gates. Each section has
`status`, `evidence_paths`, and a machine-readable failure list.

`systems_report.json` records warmup policy, CUDA synchronization policy,
candidate batch size, candidate shard id, GPU model, GPU count, memory
allocated/reserved, base model time, `Qx` time, lazy delta time, scoring time,
setup time, output tokens/sec, prompts/sec, candidates/sec, and per-GPU
aggregation fields when applicable.

Lazy top-K ensembles are evaluated by Optimus-native sample-level evaluation in
v1. LightEval is the trusted lane for base models, single-candidate
materialized exports, merged checkpoints, and distilled ensemble artifacts. A
future LightEval adapter may call the lazy ensemble runtime directly, but until
that exists LightEval results must not be presented as lazy-ensemble
confirmation.

## Correctness Tests And Acceptance Gates

Mathematical tests:

- Projection covariance: with activation rows `X` shaped `[tokens, d]`, basis
  `Q` shaped `[r, d]`, and Gaussian field `G` shaped `[m, r]`, form
  `Y = X Q.T G.T` shaped `[tokens, m]`. For each output coordinate, empirical
  covariance across calibration tokens must match `sigma_w^2 X Q.T Q X.T`
  within tolerance.
- Full-rank equivalence: if `Q` spans the input dimension, lazy perturbation
  matches dense Gaussian effect law.
- Projected-dense rank law: under `projected-dense`, measured perturbation RMS
  grows with `sqrt(H_s(r))`.
- Relative-RMS rank invariance: under `relative-output-rms`, measured relative
  output RMS is invariant to rank within tolerance.
- Budget invariance: normalized budgets do not silently increase total squared
  budget when targets are added.
- `rho=0` and `rank=0` reproduce base logits within deterministic dtype
  tolerance.

RNG and replay tests:

- `gaussian_hash_v1` golden values match on CPU, GPU, process restart, candidate
  permutation, and candidate batch-size changes.
- Candidate replay is stable across process restarts.
- Candidate replay is stable across candidate batch sizes.
- Candidate replay is stable under candidate permutation.
- Antithetic signs are exact negatives at the lazy-delta level.
- Random-field values do not depend on row order or scheduler order.

Routing and cache tests:

- Candidate-block evaluation matches one-candidate-at-a-time evaluation.
- Ragged prompt lengths do not corrupt candidate routing.
- Mixed prefill/decode scheduling does not corrupt candidate routing.
- Row-order scramble fails unless explicit `row_candidate_id` is used.
- Same prompt under different candidate ids does not share perturbed KV-cache
  state.
- Prefix caching is either disabled or candidate-keyed.

Ensemble tests:

- K=1 ensemble equals the selected single candidate.
- Candidate order permutation does not change majority-vote result except for
  documented tie-breaking.
- Ensemble artifacts replay exactly from top-K seeds and basis hash.
- Diversity metrics are reported for every top-K artifact.

Systems tests:

- Lazy hot path does not import or call adapter loading, `LoRARequest`,
  `save_seed_adapter`, or LoRA factor helpers.
- Reference backend and vLLM lazy backend match logits on fixed probes.
- Torch grouped-BMM and optimized kernels match before optimized kernels become
  default.
- vLLM runtime adapter is version-pinned or otherwise guarded by compatibility
  tests.
- Quantized search, if enabled, is labeled experimental and does not claim
  lazy/materialized parity.

## Required Metrics

Run summaries must report:

- candidates/sec;
- prompts/sec;
- output tokens/sec;
- base model time, measured with CUDA synchronization around base forward
  sections after warmup;
- `Qx` time, measured around projection kernels;
- lazy delta time, measured around random-field generation/application and
  output add;
- scoring time, excluding model forward;
- setup time, excluding model load unless explicitly reported separately;
- lazy overhead percent;
- GPU memory allocated/reserved;
- candidate batch size;
- prefix-cache policy;
- top-K ensemble cost multiplier;
- screen score, holdout score, and screen-to-holdout drop;
- diversity metrics;
- random-Q and shuffled-Q control results;
- antithetic odd/even diagnostics;
- optional gradient-capture ratio.

## Review Checklist

Before accepting an implementation PR:

- CLI uses `--basis-rank`, `--rho-grid`, `--sigma-w-grid`, `--scale-mode`, and
  `--budget-policy` with unambiguous semantics.
- Search-time method is `subspace`; adapter-backed LoRA is absent from the hot
  path.
- Target preset is `transformer-linears`, not `full-model`.
- Bases are attached to activation sites, not duplicated per target module.
- Candidate random fields are pure functions of candidate identity and indices.
- Every activation row carries explicit candidate id metadata.
- Prefix caching is disabled or candidate-keyed during search and ensemble
  serving.
- `projected-dense` and `relative-output-rms` both pass scale tests.
- Random-Q and shuffled-Q gating controls are present.
- Top-K lazy ensemble is a first-class artifact and runtime path.
- Ensemble diversity metrics are reported.
- Materialized export is optional and is not treated as the primary algorithm
  output.
- Holdout is not used for candidate selection unless explicitly reported as
  validation tuning.
- Positive winners have drift diagnostics before any dense-RandOpt
  approximation claim.
- Validation reports separate selector quality, holdout quality, top-K ensemble
  quality, and throughput.

## Final Scientific Claim

The correct v1 claim is:

> Optimus subspace lazy search evaluates radius-controlled, activation-site
> projected RandOpt candidates over transformer linear targets. In
> `projected-dense` mode, it samples the dense Gaussian output-effect law
> restricted to measured activation subspaces. In `relative-output-rms` mode, it
> compares ranks, target presets, and basis families at fixed relative
> module-output radius and fixed global budget. The primary serving artifact is
> a top-K lazy ensemble; single-candidate materialization is optional replay or
> distillation infrastructure.

The first scientific gate is not throughput. The first gate is whether
activation-SVD bases beat random orthonormal bases of equal rank at fixed
radius, fixed budget, fixed population, and fixed top-K evaluation.
