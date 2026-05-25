# Subspace Implementation Roadmap

This roadmap is the execution checklist for
`docs/full_model_lazy_kernel_design.md`. Do not start optimized kernel work
until the early correctness and basis-quality gates pass.

Current saved state, 2026-05-25:

- This roadmap is planning-only. It authorizes documentation, test planning,
  and compatibility audits, but not lazy-kernel implementation work.
- The first implementation checkpoint after this roadmap removed the known
  generated-plan legacy wrappers, added subspace-specific validation contracts,
  and made staged search fail closed until a final public route exists. Future
  implementation must keep those gates intact before touching vLLM hooks, basis
  capture, or optimized kernels.
- `optimus search` and `optimus bench` are the final public command families.
  Any runnable plan, launcher, release gate, or validation contract that still
  depends on `peft-search`, `vllm-search`, `vllm-bench`, `vllm-halving`,
  adapter hot-path flags, or old LoRA-shaped artifacts is a Phase 0 blocker.
- The public method name is `subspace`. The backend selector is `--backend`.
  Public API and artifacts must not use `activation_subspace`,
  `lazy-subspace`, `engine`, `family_state`, or LoRA adapter terminology for
  the lazy-kernel path.
- Independent review at commit `54837a2` did not accept the current
  implementation substrate. Treat the findings below as the implementation
  restart queue. Do not start basis capture, vLLM hooks, or lazy kernels until
  these Phase 0 blockers are resolved and independently re-reviewed:
  - `optimus search --help` and `optimus bench --help` do not expose the final
    subspace flags, even though generated plans and docs already depend on
    them.
  - `optimus search --backend vllm --method subspace` is generated and
    documented as the production route, but still fails closed as
    unimplemented. That is acceptable only while the plan labels it explicitly
    as a planned route, not as runnable production behavior.
  - The executable validation contract is weaker than the artifact contract:
    common provenance fields, scale grids, replay hashes, candidate identity
    consistency, top-K replay fields, and validation-report pass/failure
    semantics must be enforced.
  - `optimus systems-report` is still LoRA-shaped and does not produce a
    subspace `systems_report.json`, while release and validation gates require
    one.
  - GPU-suite launchers must call `validate-run --strict` for production
    validation; a validation JSON with `"pass": false` must make the launcher
    fail.
  - vLLM compatibility must be pinned or guarded tightly enough that a resolver
    drift cannot silently change the runtime substrate.
  - Legacy subspace aliases and compatibility parsers must be quarantined under
    private `legacy_` names or deleted; they must not remain public core API.
  - Stable subspace schemas must include the shard metadata needed for later
    one-worker-per-GPU scaling before Phase 1 is considered complete.
- The next checkpoint addressed this restart queue by exposing the final
  subspace flags in `search`/`bench` help, hardening the executable artifact
  validator, making suite validation strict from the launcher, pinning the
  default vLLM dependency, failing `systems-report` closed for unmeasured
  subspace runs, removing public legacy subspace alias normalization, and adding
  shard metadata to the public `SubspaceCandidate` schema. Phase 1 still
  requires independent re-review before runtime basis capture starts.
- The latest pre-implementation checkpoint is commit `aadc7a4`
  (`Close subspace review contract gaps`). This roadmap update records the
  remaining review queue only; it does not authorize runtime implementation.

## Next Pre-Implementation Review Queue

Before Phase 1 starts, resolve and re-review the remaining Phase 0 contract
gaps. These are roadmap blockers, not permission to start basis capture, vLLM
hooks, lazy kernels, or optimized kernels.

1. Artifact provenance:
   - `subspace_state_summary.json`, `top_k_ensemble.json`,
     `validation_report.json`, and `systems_report.json` must all carry the
     common provenance envelope: schema version, creation time, Optimus
     version, git commit/dirty state, command, environment, model revision,
     tokenizer hash, task config hash, prompt contract hash, screen split hash,
     holdout split hash, and decode config hash.
   - Artifacts that inherit provenance only through a parent summary hash must
     declare that parent explicitly; otherwise they must include the envelope
     directly.
2. Scientific gate metadata:
   - `validation_report.json` must distinguish the selected basis from
     controls with `basis_kind`, `control_basis_kinds`, `comparison`,
     `gate_type`, `epsilon`, confidence interval, locked config hash,
     selection-rule hash, primary metric, and multiple-comparison correction.
   - The reference smoke gate uses non-inferiority against random controls;
     the production gate requires a positive paired-bootstrap lower bound
     unless explicitly labeled as an engineering proceed/no-scientific-win
     exception.
3. Candidate and score uniqueness:
   - `candidates.jsonl` must reject any duplicate `candidate_id`, even if the
     duplicate row is byte-identical.
   - `candidate_scores.jsonl` must reject duplicate score rows for the same
     candidate, split, scorer, prompt/sample-set hash, and decode config hash.
4. Systems metrics:
   - `systems_report.json` numeric fields must be actual JSON numbers, not
     string-coercible values.
   - Aggregation must preserve measured fields only; missing subspace systems
     evidence is a release blocker, not a pass.
5. Public surface guards:
   - `optimus search` must reject hidden adapter-era passthrough flags such as
     activation-state prompts, family-state files, staged prompts, survivor
     counts, batch-size grids, and prompt-count grids.
   - `optimus run-plan` and `optimus run-suite` with `--method subspace` must
     reject LoRA-only knobs including `--rank`, `--sigma`, `--targets`,
     `--chunk-adapters`, `--max-loras`, `--max-cpu-loras`,
     `--keep-adapters`, and `--bench-adapters`.
6. Release checks:
   - Public-doc leakage scans must include every core source-of-truth doc:
     `README.md`, `api.md`, `optimus_design.md`,
     `full_model_lazy_kernel_design.md`, `subspace_implementation_roadmap.md`,
     `gpu_suite.md`, `prime_gpu_runbook.md`, `release_checklist.md`, and
     `evaluation.md`.
   - Prime GPU cleanup must be checked per ledger entry. A terminated entry is
     not complete unless that same entry records zero active pods or an
     equivalent cleanup verification.
7. Documentation wording:
   - Docs may list vLLM subspace search and bench as planned fail-closed
     routes, but not as implemented supported workflows until Phase 5/6 gates
     pass.
   - Any mention of adapter serving in the subspace path must be framed as a
     legacy baseline or optional materialized-export/debug path, never as the
     search hot path.

## Implementation Restart Plan

This is the saved plan for the next implementation pass. Execute in order, and
commit after each coherent slice.

1. Public surface reconciliation:
   - add final subspace options to `optimus search --help` and
     `optimus bench --help`;
   - keep unsupported runtime routes fail-closed with a roadmap pointer;
   - make generated plans label `--backend vllm --method subspace` as a planned
     route until Phase 5 lands.
2. Validation hardening:
   - require common provenance in all subspace JSON artifacts;
   - require `rho_grid` or `sigma_w_grid`, scorer/sample hashes, decode config
     hash, and deterministic RNG version;
   - require `top_k_ensemble.json` replay hashes for `subspace_state.pt`,
     `candidate_scores.jsonl`, and basis collection config;
   - validate candidate signs, radius fields, shard fields, and identity
     consistency against summary-level hashes;
   - require every validation-report section to be a real pass with nonempty
     existing evidence paths and empty failures.
3. Launcher and release fail-closed behavior:
   - make supported launchers invoke strict validation;
   - make release checks distinguish missing planned artifacts from passing
     artifacts;
   - reject subspace runs that emit LoRA-shaped systems or scoring artifacts.
4. Systems-report boundary:
   - either implement subspace `systems_report.json` aggregation from real
     subspace run artifacts, or fail closed with a clear Phase 6 blocker;
   - do not synthesize throughput metrics that were not measured.
5. Legacy quarantine:
   - delete public `activation_subspace` and `lazy_subspace` normalization for
     new code paths, or move it to explicitly private compatibility readers;
   - update tests so compatibility behavior is not mistaken for public API.
6. Schema completion:
   - add candidate shard metadata to `SubspaceCandidate` and artifacts:
     shard id, population range, worker id, device id, and immutable
     prompt/scoring config hash;
   - keep candidate identity stable before multi-GPU launch code exists.
7. Independent review gate:
   - rerun the same subagent review axes after the above changes;
   - only then start Phase 1/2 runtime implementation work.

## Phase 0: Stop-The-Line Legacy Surface Cleanup

Goal: make the public repo prove the new subspace contract before adding more
runtime code.

This phase is complete only when the runnable commands, generated plans,
launchers, tests, release checks, and validation contracts all point at the
same final API. Passing tests that still validate old LoRA adapter workflows do
not count as subspace readiness.

Deliverables:

- Update `optimus run-plan` and `optimus run-suite` so generated specs use:
  - `optimus search --backend vllm --method lora` only for explicit legacy LoRA
    baselines;
  - `optimus search --backend vllm --method subspace` only as a fail-closed
    planned route until Phase 5 lands;
  - `optimus bench --backend vllm --method lora` for legacy adapter throughput
    baselines;
  - no removed top-level commands in generated JSON or shell scripts.
- Add `run-plan` fields for the final subspace surface:
  - `--backend`;
  - `--method`;
  - `--basis-rank`;
  - `--layers`;
  - `--basis-centering`;
  - `--basis-token-source`;
  - `--scale-mode`;
  - `--rho-grid`;
  - `--sigma-w-grid`;
  - `--budget-policy`;
  - `--basis-kind`;
  - `--basis-prompts`;
  - `--target-preset`;
  - `--top-k-grid`;
  - `--candidate-batch-size`;
  - `--kernel`.
- Remove or disable staged-search emission until a final public staged-search
  route exists. Do not generate `optimus vllm-halving`.
- Update `scripts/run_optimus_gpu_suite.sh`, `scripts/README.md`, remote smoke
  scripts, and Prime runbooks so supported launchers do not pass subspace work
  through `--rank`, `--sigma`, `--chunk-adapters`, `--max-loras`, or old
  top-level command names.
- Make `perturbation-panel --method subspace` use the new subspace scale names
  or fail closed. It must not expose subspace through old `--rank` and `--sigma`
  semantics.
- Update release checks so they scan:
  - public docs;
  - `optimus/cli.py`;
  - run-plan generated JSON;
  - supported shell launchers;
  - validation contracts;
  - public commands that can emit artifacts.
- Update validation contracts so subspace runs require the subspace artifact
  shape, not LoRA adapter artifacts.
- Update old evidence docs that still describe P1024/P4096 results as
  `candidate_summary.jsonl`, adapter rows, or per-prompt LoRA artifacts. Keep
  those as legacy baselines only when explicitly labeled.

Acceptance gate:

- `optimus --help`, `optimus search --help`, `optimus bench --help`,
  `optimus run-plan --help`, and supported script docs expose only the final
  public surface for new subspace work.
- `optimus run-plan --method subspace --backend vllm ...` succeeds as a
  planning command and emits final command names, or fails closed with a
  roadmap-specific message. It must never emit removed commands.
- `optimus run-plan` default LoRA baseline plans use final `search`/`bench`
  routes, not removed wrappers.
- Validation has separate contracts for:
  - legacy LoRA adapter baselines;
  - subspace reference runs;
  - subspace vLLM runs;
  - systems reports.
- Release checks fail if any supported launcher, generated plan, public docs, or
  validation gate promotes removed wrappers or LoRA-shaped subspace artifacts.
- Narrow tests prove old wrappers are unknown, generated plans contain no
  removed command names, and subspace validation requires
  `subspace_state.pt`, `candidate_scores.jsonl`, `top_k_ensemble.json`,
  `validation_report.json`, and `systems_report.json`.

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
- Run-plan and run-suite do not emit removed wrappers.

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
- Move old LoRA-only validation and systems-report assumptions behind explicit
  legacy-baseline names. They must not be the default proof path for subspace.

Acceptance gate:

- `rg` guards over public docs, CLI help, summaries, and subspace packages prove
  no public `activation_subspace`, `lazy-subspace`, `family_state`,
  `vllm-search`, `peft-search`, `vllm-halving`, `vllm-bench`, or adapter-hot-path
  names remain.
- Any remaining legacy parser is private, covered by compatibility tests, and
  cannot be reached by new subspace runs.

## Phase 0C: Executable Artifact Contracts

Goal: make schemas test-enforceable before implementation starts.

Deliverables:

- Add explicit run-contract families for subspace:
  - `subspace_reference_search`;
  - `subspace_vllm_search`;
  - `subspace_lazy_ensemble`;
  - `subspace_systems_report`.
- Validate required files:
  - `subspace_state.pt`;
  - `subspace_state_summary.json`;
  - `candidates.jsonl`;
  - `candidate_scores.jsonl`;
  - `top_k_ensemble.json`;
  - `summary.json`;
  - `validation_report.json`;
  - `systems_report.json`.
- Validate replay-critical fields in JSON artifacts:
  - `schema_version`;
  - `backend`;
  - `method`;
  - `basis_hash`;
  - `target_set_hash`;
  - `scale_mode`;
  - `rho_grid` or `sigma_w_grid`;
  - `budget_policy`;
  - `rng_version`;
  - `candidate_routing`;
  - `prefix_cache_policy`;
  - scorer version;
  - prompt/sample-set hashes;
  - decode config hash.
- Validate `top_k_ensemble.json` contains full candidate identities, not only
  candidate ids.
- Add `gaussian_hash_v1` golden-vector tests before any random-field backend is
  accepted.

Acceptance gate:

- A synthetic subspace run fixture can pass validation with only the documented
  artifacts and required fields.
- A LoRA adapter run fixture cannot pass as a subspace run.
- Missing `top_k_ensemble.json`, missing full candidate identities, missing
  `rng_version`, missing scorer/sample hashes, or missing systems metrics fail
  validation.
- Golden-vector RNG tests cover process restart, candidate permutation,
  candidate batch-size changes, antithetic signs, and row-order independence.

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
- Publish `gaussian_hash_v1` test vectors and keep the exact byte payload
  stable unless `rng_version` changes.
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
- Golden-vector tests prove `gaussian_hash_v1` is stable across Python
  versions, worker counts, row order, and candidate-block composition.
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
- Activation-SVD passes a predeclared non-inferiority test against both random
  controls on the locked primary metric: paired bootstrap 95% CI lower bound for
  `activation_svd - best_control` must be at least `-epsilon`, where `epsilon`
  is written in the run config before scoring starts. This gate only authorizes
  vLLM parity work; it is not the production scientific claim.

## Phase 5: vLLM Eager Wrapper

Goal: integrate subspace search with vLLM without forking vLLM.

Deliverables:

- `optimus.backends.vllm_subspace` eager wrapper around vLLM model execution.
- vLLM version pin/compatibility guard before runtime enablement.
- Fail-closed feature detection for model-runner hooks, request metadata, and
  custom-op registration points.
- Explicit `row_candidate_id` routing into perturbed target modules.
- Request metadata propagation via the selected vLLM custom-arguments path.
- Concrete row-routing descriptor propagated from request metadata to the model
  runner:
  - `request_id`;
  - `sequence_id`;
  - prefill/decode phase;
  - flattened token row start/count;
  - position ids or equivalent row-position mapping;
  - `candidate_id`;
  - candidate slot within the current candidate block;
  - basis/runtime config hash.
- Prefix cache policy:
  - v1 supports only `disabled-for-search`;
  - no candidate-different KV-cache sharing;
  - candidate-keyed prefix caching is a later PR and must not be enabled until
    vLLM cache-key integration has explicit tests.
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

- Primary metric is predeclared as top-K ensemble holdout score at one locked
  configuration: fixed K, rank, radius, target preset, task panel, seed panel,
  scorer config, and aggregation rule.
- Rank/radius/K grids are calibration or validation searches. If a grid is used
  in the gate, the selection rule must be screen-only and identical across basis
  families, or the analysis must apply the predeclared multiple-comparison
  correction from the design doc.
- Activation-SVD must beat random-orthonormal and shuffled-SVD controls with a
  paired bootstrap 95% CI lower bound above zero on the locked primary metric.
- A statistically indistinguishable tie can proceed only if a predeclared
  engineering review accepts the label
  `engineering_proceed_no_scientific_win` and shows at least one operational
  advantage at equal quality:
  - at least 25% lower logit-KL/drift;
  - at least 20% lower lazy overhead;
  - at least 10 percentage points higher captured activation energy.
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

0. Fix generated run specs, supported launchers, release checks, and validation
   contracts so they no longer bless the old LoRA adapter suite as the subspace
   path.
1. Public `search`/`bench` router and old-wrapper removal, including run-plan
   and run-suite.
2. Public-name guards in tests and release checks.
3. Subspace artifact contract fixtures and validation tests.
4. `optimus.subspace` schema and artifact writers.
5. Target preset registry and activation-site specs.
6. Deterministic `gaussian_hash_v1` with golden vectors.
7. Basis capture and control bases.
8. Scale resolver and budget policies.
9. Reference evaluator with top-K artifacts.
10. Evaluation harness contract and sample-level details.
11. vLLM one-target wrapper with fail-closed compatibility guard.
12. vLLM routing/cache adversarial tests.
13. Full target presets over vLLM.
14. Production scientific gate.
15. Materialized export and distillation artifacts.
16. Optimized kernel only after the scientific and p128 speed gates pass.
