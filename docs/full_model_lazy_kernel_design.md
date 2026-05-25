# Activation-Site Projected RandOpt Lazy Kernel

## Status And Scope

This document is the implementation source of truth for the planned Optimus
transformer-linear subspace RandOpt backend over vLLM. The public method name is
`subspace`. The precise mechanism is activation-site projected RandOpt: random
perturbations are applied to transformer block linear modules through activation
bases captured at the sites those modules read from.

Current repository status: this is a design and artifact contract. The
production vLLM subspace backend is not implemented yet, and public vLLM
subspace search/bench routes must fail closed until the roadmap Phase 5/6 gates
land. Existing LoRA adapter serving code is legacy baseline infrastructure and
is not the subspace search hot path.

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
`Delta W_t,c = beta_t G_t,c Q_s(t)`, but the planned search-time execution is
lazy: the vLLM backend will inject `beta_t G_t,c Q_s(t) x` inside model
execution and must never load, swap, or materialize per-candidate adapters in
the hot path.

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

Every activation site must be represented by the public `ActivationSite` schema
before it can be used by a backend. Required fields are:

- `site_id`;
- `architecture_family`, for example `qwen3_text` or `llama_decoder`;
- `layer_index` and `block_path`;
- `read_tensor_path` and `hook_point`;
- `norm_position`;
- `shape_convention`, with `[tokens, hidden]` after sequence flattening in v1;
- `runtime_dtype` and `accumulation_dtype`;
- `tensor_parallel_sharding_policy`;
- `target_module_ids`;
- `calibration_prompt_ids_hash` and `calibration_decode_config_hash`;
- `basis_control_seed`;
- `transductive`;
- `input_dim`;
- `basis_kind`;
- `requested_rank` and `effective_rank`;
- `basis_tensor_key` and `basis_tensor_sha256`;
- `singular_values`;
- `captured_energy`, `prefill_captured_energy`, and
  `decode_captured_energy`;
- `H_s` and `A_s`;
- `orthonormality_error` and `gram_error`;
- `num_calibration_tokens`.

The v1 Qwen/Llama mapping is: `attn_in` is the normalized attention input read
by q/k/v projections, `o_in` is the attention output-projection input after
attention and before `o_proj`, `mlp_in` is the normalized MLP input read by
gate/up projections, and `down_in` is the post-activation gated MLP tensor read
by `down_proj`. Any architecture that does not expose these tensors with the
same semantics needs an explicit mapping entry and tests before it can use the
`transformer-linears` preset.

The default full transformer preset is `transformer-linears`. It means selected
dense transformer block linear maps. It excludes embeddings, LM head, layer
norms, routers, and architecture-specific nonstandard modules. V1 public target
selection is `--target-preset` plus `--layers` only. The artifact field
`explicit_targets` is reserved for internal metadata and future target-manifest
work; it is not a public ad hoc target-selection flag in v1.

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
  "scale_mode": "relative-output-rms",
  "rho_or_sigma_w": 0.01,
  "budget_policy": "per-block-equal",
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
  "aggregation": "majority-vote | mean-logprob | score-sum",
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
      "scale_mode": "relative-output-rms",
      "rho_or_sigma_w": 0.01,
      "budget_policy": "per-block-equal",
      "budget_hash": "...",
      "rng_version": "gaussian_hash_v1",
      "runtime_dtype": "bf16"
    }
  ],
  "basis_hash": "...",
  "scale_mode": "relative-output-rms",
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
cannot assume shared decode trajectories. V1 disables prefix caching for
subspace search and ensemble serving; candidate-keyed cache reuse is a later
integration problem, not a v1 optimization.

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
predeclared operational advantage at equal quality using the metric contract in
`drift_diagnostics`: at least 25% lower `logit_kl_mean`, at least 25% lower
`hidden_state_rms_drift`, at least 20% lower lazy overhead, or at least 10
percentage points higher captured activation energy. Secondary metrics are
reported for diagnosis only: improvement density, best single-candidate holdout
score, diversity-adjusted ensemble gain, and screen-to-holdout drop.

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

The v1 row-routing descriptor is explicit and scheduler-order independent. At
the vLLM request boundary, Optimus attaches candidate metadata to each logical
request. At the model-runner boundary, Optimus constructs a flattened
row-routing tensor/table with:

- request id;
- sequence id;
- prefill or decode phase;
- flattened token-row start and count;
- position ids or equivalent row-position mapping;
- stable candidate id;
- candidate slot within the current candidate block;
- basis/runtime config hash.

Chunked prefill, decode-only steps, ragged prompts, and row-order changes must
preserve this mapping. Kernel fast paths may reorder rows internally only if the
pre-reorder and post-reorder descriptors round-trip in routing tests.

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
targets when they occur before or inside the cached computation. V1 therefore
has exactly one supported policy:

Default v1:

```bash
--prefix-cache-policy disabled-for-search
```

Candidate-different requests must not share KV-cache or prefix-cache entries in
v1. A future `candidate-keyed` policy is a separate PR and non-goal for v1. It
must define the exact vLLM cache-key integration, including use of cache salt or
extra-hash fields where applicable, and must key every computation-affecting
field: base model hash, basis hash, candidate id, target set hash, scale mode,
radius, budget hash, RNG version, runtime dtype, target preset, and runtime
config hash.

V1 exact parity claims require unquantized fp16/bf16 base weights. Quantized
base search may be allowed experimentally, but must report base weight dtype,
runtime dtype, lazy delta dtype, quantization kind, and materialization policy.
A bf16 lazy delta added to a quantized base linear is not automatically
equivalent to materializing and quantizing `W + Delta W`.

## Public CLI

Planned canonical search command. In the current repository this route is
fail-closed and exits with a roadmap pointer; it becomes runnable only after the
Phase 5 vLLM backend gate passes.

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

Projected-dense science command. This has the same current fail-closed status
for the vLLM backend.

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
| `subspace_state.pt` | Versioned activation-site basis tensors referenced by the summary metadata. |
| `subspace_state_summary.json` | Validated activation-site basis, provenance, and scale metadata. |
| `candidates.jsonl` | Candidate identities before or during scoring. |
| `candidate_scores.jsonl` | Per-candidate screen and selected holdout scores. |
| `top_k_ensemble.json` | Primary top-K lazy ensemble artifact. |
| `summary.json` | Run-level summary. |
| `validation_report.json` | Gating, controls, bootstrap, diversity, and drift reports. |
| `systems_report.json` | Throughput and memory report. |
| `exports/` | Optional single-winner or distillation artifacts only. |

All JSON artifacts include the provenance envelope directly: `schema_version`,
`created_at`, `optimus_version`, `git_commit`, `git_dirty`, `command`,
`environment`, `model_id_or_path`, `model_revision`, `tokenizer_hash`,
`task_config_hash`, `decode_config_hash`, `prompt_contract_hash`,
`screen_split_hash`, and `holdout_split_hash`. V1 does not support inherited
provenance through parent-summary references; a JSON artifact that omits this
envelope is invalid.

`subspace_state_summary.json` required fields:

```json
{
  "schema_version": "subspace_state_v1",
  "created_at": "2026-05-25T00:00:00Z",
  "optimus_version": "0.1.0",
  "git_commit": "...",
  "git_dirty": false,
  "command": ["optimus", "search", "..."],
  "environment": {"python": "...", "cuda": "..."},
  "model_id_or_path": "...",
  "model_revision": "...",
  "tokenizer_hash": "...",
  "task_config_hash": "...",
  "decode_config_hash": "...",
  "prompt_contract_hash": "...",
  "screen_split_hash": "...",
  "holdout_split_hash": "...",
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
      "architecture_family": "qwen3_text",
      "layer_index": 17,
      "block_path": "model.layers.17",
      "read_tensor_path": "model.layers.17.input_layernorm.output",
      "hook_point": "forward_output",
      "norm_position": "post_input_norm",
      "shape_convention": "[tokens, hidden]",
      "runtime_dtype": "bf16",
      "accumulation_dtype": "fp32",
      "tensor_parallel_sharding_policy": "replicated",
      "target_module_ids": [
        "layer_17.self_attn.q_proj",
        "layer_17.self_attn.k_proj",
        "layer_17.self_attn.v_proj"
      ],
      "calibration_prompt_ids_hash": "...",
      "calibration_decode_config_hash": "...",
      "basis_control_seed": 1234,
      "transductive": false,
      "input_dim": 4096,
      "basis_kind": "activation-svd",
      "requested_rank": 128,
      "effective_rank": 128,
      "basis_tensor_key": "...",
      "basis_tensor_sha256": "...",
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

`subspace_state.pt` is the tensor payload. Until the Phase 1 writer/reader
lands, validators require a loadable payload with schema
`subspace_state_payload_v1`, a `basis_tensors` map keyed by each
`basis_tensor_key`, and per-basis tensor hashes matching
`basis_tensor_sha256`. They also recompute the file digest and compare it to
`summary.json.subspace_state_hash`.

`summary.json` required fields:

```json
{
  "schema_version": "subspace_run_summary_v1",
  "kind": "subspace_vllm_search",
  "backend": "vllm",
  "method": "subspace",
  "created_at": "...",
  "optimus_version": "...",
  "git_commit": "...",
  "git_dirty": false,
  "command": ["optimus", "search", "..."],
  "environment": {"python": "...", "cuda": "..."},
  "model_id_or_path": "Qwen/Qwen3-4B",
  "model_revision": "...",
  "tokenizer_hash": "...",
  "task_config_hash": "...",
  "prompt_contract_hash": "...",
  "screen_split_hash": "...",
  "holdout_split_hash": "...",
  "screen_holdout_overlap": 0,
  "population": 1024,
  "scale_mode": "relative-output-rms",
  "rho_grid": [0.002, 0.005, 0.01, 0.02],
  "sigma_w_grid": null,
  "budget_policy": "per-block-equal",
  "basis_hash": "...",
  "target_set_hash": "...",
  "basis_collection_config_hash": "...",
  "subspace_state_hash": "...",
  "candidate_scores_hash": "...",
  "resolved_target_scales": [
    {
      "target_id": "layer_17.self_attn.q_proj",
      "budget_weight": 0.0714,
      "beta_t_by_radius": {"0.01": 0.000083}
    }
  ],
  "rng_version": "gaussian_hash_v1",
  "candidate_routing": "row_candidate_id",
  "prefix_cache_policy": "disabled-for-search",
  "scorer_name": "...",
  "scorer_version": "...",
  "prompt_ids_hash": "...",
  "sample_set_hash": "...",
  "prompt_scoring_config_hash": "...",
  "decode_config_hash": "...",
  "kernel": "torch",
  "candidates_per_sec": 12.3,
  "prompts_per_sec": 45.6,
  "output_tokens_per_sec": 789.0,
  "lazy_overhead_pct": 0.18
}
```

`screen_holdout_overlap` is a required integer and must be `0` for strict
subspace validation. Throughput fields in `summary.json` mirror measured
systems evidence; they must be JSON numbers, not string-coercible values.

`candidates.jsonl` rows contain the complete `SubspaceCandidate` identity:
`candidate_id`, `direction_seed`, `sign`, `basis_hash`, `target_set_hash`,
`scale_mode`, `rho_or_sigma_w`, `budget_policy`, `budget_hash`, `rng_version`,
`runtime_dtype`, `radius_index`, `target_preset`, `basis_rank`, `shard_id`,
`shard_population_start`, `shard_population_end`, `worker_id`, `device_id`,
and `prompt_scoring_config_hash`. `sign` is serialized as `"+"` or `"-"` in
artifacts; kernels may resolve that to `+1` or `-1` internally.

`candidate_scores.jsonl` rows contain `candidate_id`, split,
`selection_stage`, `selection_rule_hash`, `promoted_by_candidate_id`, scorer
name and version, aggregate metrics, sample count, prompt ids hash, sample-set
hash, decode config hash, elapsed time, token counts, and optional path to
per-sample rows. Selected holdout rows must include the selector that promoted
the candidate through `selection_rule_hash` and `promoted_by_candidate_id`; a
holdout score row without this provenance is invalid.

`top_k_ensemble.json` contains the full candidate identities, not only ids, plus
`aggregation`, `tie_break_policy`, `selection_rule`, `K`, `scorer_version`,
`prompt_ids_hash`, `sample_set_hash`, `prompt_scoring_config_hash`,
`basis_collection_config_hash`, `runtime_config_hash`, `decode_config_hash`,
`rng_version`, and replay hashes for `subspace_state.pt` and
`candidate_scores.jsonl`. The top-level replay hashes must match the
corresponding run-summary hashes.

`validation_report.json` has separate sections for math tests, RNG/replay tests,
routing/cache tests, selector quality, holdout quality, ensemble quality, drift
diagnostics, random/shuffled controls, throughput gates, and a
`scientific_gate_contract`. Each section has `status`, `evidence_paths`, and a
machine-readable failure list. A completed run passes validation only when every
required section has `status: "pass"`, nonempty evidence paths that exist in the
run directory, and an empty failure list. Evidence paths must point to
section-specific JSON evidence, not back to `summary.json` or
`validation_report.json`.

Each evidence file must use this minimal schema:

```json
{
  "evidence_schema_version": "validation_evidence_v1",
  "section": "math_tests",
  "status": "pass",
  "generated_at": "2026-05-25T00:00:00Z",
  "command": ["python", "-m", "pytest", "..."],
  "checks": [{"name": "projection_covariance", "passed": true}],
  "metrics": {"max_error": 0.00001},
  "artifacts": ["math_projection_covariance.json"]
}
```

At least one of `checks`, `metrics`, or `artifacts` must be nonempty. A bare
`{"section": "...", "status": "pass"}` marker is invalid.

The scientific gate section also records
`locked_config_hash`, `selection_rule_hash`, `primary_metric`,
`multiple_comparison_correction`, locked K/rank/radius/target/aggregation
fields, `selection_split`, `holdout_tuned`, `screen_holdout_overlap`, and a
numeric confidence interval so a gate result is not merely self-attested prose.

`drift_diagnostics` has a fixed v1 metric contract. The probe split is an
immutable unlabeled prompt/token-row set identified by `probe_split_hash`, and
the reference output is the unperturbed base model under the same model
revision, tokenizer, decode config, dtype policy, and target preset. The
required logit statistic is `logit_kl_mean`: mean token-row
`KL(softmax(base_logits / T) || softmax(candidate_logits / T))` with
temperature `T=1.0` unless the evidence artifact records another value. The
required activation statistic is `hidden_state_rms_drift`: mean over reported
probe rows and sites of `||h_candidate - h_base||_2 / max(||h_base||_2, eps)`,
with `eps` recorded in the evidence artifact. Evidence must record
`probe_split_hash`, `reference_artifact_hash`, `candidate_artifact_hash`,
aggregation rule, temperature/epsilon where applicable, and sample count.

An `engineering_proceed_no_scientific_win` gate may cite only one of these
operational-advantage metrics:

| metric | required direction | minimum delta |
| --- | --- | --- |
| `logit_kl_mean_reduction_pct` | lower than best equal-quality control | `25.0` |
| `hidden_state_rms_drift_reduction_pct` | lower than best equal-quality control | `25.0` |
| `lazy_overhead_reduction_pct` | lower synchronized overhead | `20.0` |
| `captured_energy_gain_pct_points` | higher captured energy | `10.0` |

The operational-advantage record stores `metric`, `delta`, `direction`,
`probe_split_hash`, `reference_artifact_hash`, and `aggregation`.

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
  `Y = X Q.T G.T` shaped `[tokens, m]`. Estimate token-token covariance by
  averaging `Y[:, j] Y[:, j].T` over many independent random-field draws and/or
  output coordinates `j`. The estimator must match
  `sigma_w^2 X Q.T Q X.T` within a predeclared tolerance; one output coordinate
  from one draw is not sufficient evidence.
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
- Prefix caching is disabled for v1 subspace search and ensemble serving.

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
- Prefix caching is disabled during v1 search and ensemble serving.
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
