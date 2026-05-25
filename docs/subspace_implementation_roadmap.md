# Subspace Implementation Roadmap

This roadmap is the execution checklist for
`docs/full_model_lazy_kernel_design.md`. Do not start optimized kernel work
until the early correctness and basis-quality gates pass.

## Phase 0A: Public API Router

Goal: expose the final public command shape before adding more runtime code.

Deliverables:

- Add `optimus search` with `--backend vllm|transformers` and
  `--method dense|lora|subspace`.
- Add `optimus bench` for backend throughput checks.
- Remove public CLI wrappers `peft-search`, `vllm-search`, `vllm-halving`, and
  `vllm-bench`.
- Unsupported routes, especially `--backend vllm --method subspace` before
  Phase 5, must fail closed with an explicit roadmap pointer. They must never
  route through adapter materialization as a placeholder.

Acceptance gate:

- Public docs and CLI help expose only the final commands.
- Old wrappers are absent from `optimus --help` and fail as unknown commands.
- Release check includes the design doc and roadmap as required public docs.

## Phase 0B: Public Names And Legacy Quarantine

Goal: ensure the final library is not burdened by old public names while still
allowing intentionally private compatibility readers during migration.

Deliverables:

- Replace public method/family names with `subspace` and
  `subspace_gaussian_rank_r`.
- Replace public artifact names with `subspace_state.pt` and
  `subspace_state_summary.json`.
- Replace `--rank` with `--basis-rank` for subspace, `--sigma` with
  `--sigma-w-grid` or `--rho-grid`, and `--activation-state-prompts` with
  `--basis-prompts`.
- Add public-name guard tests that fail on old names in docs, CLI help,
  summaries, and new run artifacts.
- Quarantine compatibility code under private modules with explicit
  `legacy_` names, or delete it. Subspace hot paths must not import
  `LoRARequest`, `save_seed_adapter`, adapter loading, or LoRA factor helpers.
- Delete or move old experiment scripts into ignored local paths if they are not
  needed for current tests.

Acceptance gate:

- `rg` guards over public docs, CLI help, summaries, and subspace packages prove
  no public `activation_subspace`, `lazy-subspace`, `family_state`,
  `vllm-search`, `peft-search`, `vllm-halving`, `vllm-bench`, or adapter-hot-path
  names remain.
- Any remaining legacy parser is private, covered by compatibility tests, and
  cannot be reached by new subspace runs.

## Phase 1: Core Subspace Data Model

Goal: introduce activation-site state and candidate identity without touching
vLLM internals yet.

Deliverables:

- Add `optimus.subspace` with:
  - `ActivationSite`;
  - `TargetModule`;
  - `SubspaceState`;
  - `TargetScale`;
  - `CandidateRandomField`;
  - `SubspaceCandidate`;
  - `SubspaceEnsemble`.
- Implement `subspace_state.pt` read/write with schema
  `subspace_state_v1`.
- Implement target presets `qv`, `attn-qkvo`, `mlp`, and
  `transformer-linears`.
- Implement activation-site sharing:
  - q/k/v share `attn_in`;
  - o uses `o_in`;
  - gate/up share `mlp_in`;
  - down uses `down_in`.
- Implement deterministic stateless Gaussian random-field generation with
  process-, batch-, and scheduler-independent replay.
- Define candidate shard metadata in the schema now:
  - shard id;
  - shard population range;
  - worker id;
  - device id;
  - immutable prompt/scoring config hash.
  Later multi-GPU work may add launch/supervision, but must not change these
  fields.

Acceptance gate:

- Unit tests show candidate random fields replay across process restarts,
  candidate batch sizes, and candidate permutations.
- Unit tests show target modules that share an activation site do not duplicate
  basis tensors.

## Phase 2: Basis Builder And Scale Resolver

Goal: build calibrated activation-site bases and unambiguous perturbation
scales.

Deliverables:

- Basis kinds:
  - `activation-svd`;
  - `random-orthonormal`;
  - `shuffled-activation-svd`.
- Basis centering:
  - `none`;
  - `mean`.
- Basis token source:
  - `prefill`;
  - `decode`;
  - `prefill+decode`.
- Split hygiene metadata:
  - `train`;
  - `screen_unlabeled`;
  - `public_unlabeled`;
  - `holdout_forbidden`.
- Calibration stats:
  - `H_s`;
  - `A_s`;
  - captured energy;
  - singular values;
  - orthonormality error;
  - Gram error;
  - target output power `P_t`.
- Scale modes:
  - `projected-dense` with `--sigma-w-grid`;
  - `relative-output-rms` with `--rho-grid`.
- Budget policies:
  - `raw-dense`;
  - `per-target-equal`;
  - `per-layer-equal`;
  - `per-block-equal`;
  - `custom-json`.

Acceptance gate:

- Projection covariance and full-rank equivalence tests pass.
- `projected-dense` rank law and `relative-output-rms` rank invariance tests
  pass.
- Degenerate `H_s`, `P_t`, rank, or NaN scale stats fail closed.

## Phase 3: Reference Lazy Evaluator

Goal: validate the math and scoring loop without vLLM integration risk.

Deliverables:

- Pure PyTorch/Transformers reference evaluator for tiny models.
- Explicit `row_candidate_id` routing in the reference path.
- Candidate-block and one-candidate-at-a-time parity tests.
- Antithetic odd/even diagnostics.
- K=1 and top-K lazy ensemble reference serving.
- `candidate_scores.jsonl`, `top_k_ensemble.json`, `validation_report.json`,
  and replay metadata are written by the reference path before any scientific
  gate uses its results.

Acceptance gate:

- `rho=0` and `basis_rank=0` reproduce base logits.
- Antithetic signs are exact negatives at lazy-delta level.
- Candidate-block evaluation matches one-at-a-time evaluation.
- Top-K K=1 equals the selected single candidate.
- Top-K ensemble artifacts replay from full candidate identities, basis hash,
  prompt/scoring config hash, decode config hash, and deterministic RNG version.

## Phase 3.5: Evaluation Harness Contract

Goal: use existing evaluation harnesses where they apply and avoid rebuilding a
parallel benchmark runner.

Deliverables:

- Immutable task/scorer config schema with metric ids, scorer version, prompt
  ids, split hashes, decode config, and saved sample-level details.
- LightEval lane for base models, materialized single-candidate exports, merged
  checkpoints, and distilled ensemble artifacts.
- Optimus-native lazy-ensemble evaluator for `top_k_ensemble.json` until a
  direct LightEval runtime adapter exists.
- Adapter interface that can later expose lazy ensembles to LightEval without
  changing candidate identity or artifact schemas.

Acceptance gate:

- Every metric in `candidate_scores.jsonl` and `validation_report.json` points
  to a scorer version and sample set.
- Docs state when LightEval is authoritative and when Optimus-native lazy
  evaluation is the only valid v1 path.
- No scientific gate can use a metric that lacks sample-level details.

## Phase 4A: Reference Smoke Gate

Goal: catch math, replay, and obvious basis failures before vLLM integration.

Experiment:

```text
basis_kind in {activation-svd, random-orthonormal, shuffled-activation-svd}
scale_mode = relative-output-rms
budget_policy = per-block-equal or predeclared alternative
basis_rank = predeclared grid
rho = predeclared grid
population = fixed
seed_panel = fixed
screen_split = fixed
holdout_split = fixed
top_k_grid = fixed
```

Metrics:

- improvement probability `Pr[S(c) > S(base)]`;
- best single-candidate score;
- top-K ensemble score;
- screen-to-holdout drop;
- top-K diversity metrics;
- antithetic odd/even diagnostics;
- captured activation energy;
- optional gradient-capture ratio.

Acceptance gate:

- Reference path passes math, replay, artifact, and scorer contracts.
- Activation-SVD is not obviously worse than both random controls on the
  predeclared primary metric. This gate only authorizes vLLM parity work; it is
  not the production scientific claim.

## Phase 5: vLLM Eager Wrapper

Goal: integrate subspace search with vLLM without forking vLLM.

Deliverables:

- `optimus.backends.vllm_subspace` eager wrapper around vLLM model execution.
- vLLM version pin/compatibility guard before runtime enablement.
- Fail-closed feature detection for model-runner hooks, request metadata, and
  custom-op registration points.
- Explicit `row_candidate_id` routing into perturbed target modules.
- Request metadata propagation via the selected vLLM custom-arguments path.
- Prefix cache policy:
  - default `disabled-for-search`;
  - no candidate-different KV-cache sharing.
- Single-target proof-of-life, then read-site sharing, then
  `transformer-linears`.
- Throughput metrics:
  - candidates/sec;
  - prompts/sec;
  - output tokens/sec;
  - base model time;
  - `Qx` time;
  - lazy delta time;
  - setup time;
  - scoring time;
  - lazy overhead percent;
  - GPU memory allocated/reserved;
  - candidate batch size;
  - prefix-cache policy.

Acceptance gate:

- vLLM eager wrapper matches reference logits on fixed probes.
- vLLM import/runtime compatibility tests pass against the pinned version range.
- Ragged prompts and mixed prefill/decode scheduling preserve candidate
  routing.
- Same prompt under different candidates cannot share perturbed KV-cache state.
- p16/p32 smoke passes on a Qwen3-class model.

## Phase 5.5: Production Scientific Gate

Goal: answer whether activation-SVD bases are worth optimizing on the actual
production substrate before investing in custom kernels.

Acceptance gate:

- Primary metric is predeclared as top-K ensemble holdout score at fixed K,
  rank grid, radius grid, task panel, seed panel, and scorer config.
- Activation-SVD must beat random-orthonormal and shuffled-SVD controls with a
  paired bootstrap 95% CI lower bound above zero on the primary metric.
- A statistically indistinguishable tie can proceed only if a predeclared
  engineering review accepts materially lower drift or materially better
  throughput at equal quality.
- All secondary metrics, logit-KL/drift diagnostics, and screen-to-holdout drops
  are reported but do not replace the primary gate.
- Holdout-tuned rank/radius/K/target choices are labeled validation and require
  a fresh final test split for claims.

## Phase 6: p128 Speed And Search Gate

Goal: decide whether Torch BMM is good enough before p1024.

Matched benchmarks:

- base vLLM;
- LoRA adapter search/replay baseline;
- q/v subspace;
- `attn-qkvo` subspace;
- `mlp` subspace;
- `transformer-linears` subspace.

Run conditions:

- same model;
- same prompts;
- same decode settings;
- same scorer;
- same GPU type;
- same population where applicable.

Decision gates:

- If `Qx + lazy_delta` exceeds 25% of synchronized hot model time after warmup,
  add Triton or FlashInfer
  grouped kernels before p1024.
- If candidate setup dominates hot model time, fix candidate-block residency or
  random-field generation before p1024.
- If `transformer-linears` is more than 2x slower than target bands at matched
  quality, profile by target group and consider staged target rollout.

## Phase 7: Top-K Ensemble Runtime Hardening

Goal: harden the primary RandOpt artifact after it already exists in the
reference and vLLM gate paths.

Deliverables:

- `top_k_ensemble.json`.
- Lazy top-K serving path using the same kernel shape as search.
- Aggregation modes:
  - `majority-vote`;
  - `mean-logprob`;
  - `score-sum`.
- Task-level guardrails for open-ended structured generation.
- Diversity metrics:
  - answer disagreement;
  - logprob/logit correlation;
  - JS or symmetric KL where available;
  - distinct answer count;
  - oracle-top-K score where available;
  - marginal ensemble gain over best single candidate.

Acceptance gate:

- Ensemble artifacts replay from top-K seeds and basis hash.
- Candidate order does not change results except for documented tie-breaking.
- K=1 equals selected single candidate.

## Phase 8: Materialized Export

Goal: support replay/debug/distillation without confusing export with the
algorithm.

Deliverables:

- Optional single-candidate PEFT/vLLM-compatible export.
- Lazy-vs-materialized parity on fixed probes.
- Distillation-target export for ensembles.

Acceptance gate:

- Export is never used in the search hot path.
- Docs and summaries mark export as optional replay/distillation
  infrastructure.

## Phase 9: Optimized Kernels

Goal: optimize only after the scientific and p128 speed gates justify it.

Deliverables:

- Triton or FlashInfer grouped kernel for lazy delta.
- Parity tests against Torch path.
- Version guard for vLLM runtime integration.
- CUDA-graph compatibility assessment.

Acceptance gate:

- Optimized kernels match Torch path within dtype tolerance.
- Optimized path improves matched p128 throughput enough to justify p1024/p4096
  runs.

## Phase 10: Multi-GPU Follow-Up Boundary

Goal: keep v1 single-GPU execution while preserving the shard schema introduced
in Phase 1.

V1 worker API must already accept:

- candidate shard spec;
- vLLM worker config;
- device id;
- path/object reference for `subspace_state.pt`;
- deterministic candidate generation config;
- immutable prompt/scoring config.

Future multi-GPU PR should add only:

- process launch and worker supervision;
- candidate shard assignment;
- per-GPU metrics aggregation;
- deterministic merge/retry handling.

It must not rewrite candidate identity, basis artifacts, scoring, materialized
export, candidate shard metadata, or top-K ensemble artifacts.

## Commit-Sized PR Slices

Preferred implementation order:

1. Public `search`/`bench` router and old-wrapper removal.
2. Public-name guards in tests and release checks.
3. `optimus.subspace` schema and artifact writers.
4. Target preset registry and activation-site specs.
5. Deterministic `gaussian_hash_v1` with golden vectors.
6. Basis capture and control bases.
7. Scale resolver and budget policies.
8. Reference evaluator with top-K artifacts.
9. Evaluation harness contract and sample-level details.
10. vLLM one-target wrapper with fail-closed compatibility guard.
11. vLLM routing/cache adversarial tests.
12. Full target presets over vLLM.
13. Production scientific gate.
14. Materialized export and distillation artifacts.
15. Optimized kernel only after the scientific and p128 speed gates pass.
