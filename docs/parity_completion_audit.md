# Parity Completion Audit

## Objective

Establish an end-to-end perturbation search method that is faster than full
dense Gaussian RandOpt while matching or beating it on:

```text
stability
drift
candidate evaluation speed
convenience / replayability
robustness
quality
```

## Current Status

Not achieved.

The current evidence shows that the lab can run dense Gaussian reference search,
LoRA-family search, vLLM LoRA serving probes, geometry audits, and parity
reports. It does not yet show a LoRA-style family with full dense-Gaussian
search-utility parity.

Use the machine-readable goal audit to prevent partial evidence from being
treated as completion:

```bash
python -m randopt_lora_lab.goal_audit \
  --reproduction-audit results/PAPER_DENSE/reproduction_audit/summary.json \
  --parity-report results/PARITY/report/summary.json \
  --parity-arm lora \
  --backend-gate results/BACKEND_GATE/summary.json \
  --multirun-gate results/spectral_vllm_multirun_gate/summary.json \
  --prompt-robustness results/PROMPT_ROBUSTNESS/summary.json \
  --drift-report results/DRIFT_AUDIT/summary.json \
  --eval-validity results/SEARCH_RUN/validity/summary.json \
  --adapter-run results/VLLM_LORA_SEARCH \
  --out results/goal_audit
```

Missing evidence is a failure. The audit only passes when every objective axis
has a concrete artifact.

For spectral vLLM confirmation, aggregate across runs before making any project
claim:

```bash
python -m randopt_lora_lab.multirun_gate \
  --run results/spectral_vllm_confirmation_rank32_c1p5_seed1 \
  --run results/spectral_vllm_confirmation_rank32_c1p5_seed2 \
  --parity-arm lora \
  --min-runs 2 \
  --min-prompt-variants 2 \
  --max-zero-regret-k 8 \
  --out results/spectral_vllm_multirun_gate
```

The multi-run gate is intentionally strict: single-seed, default-prompt-only, or
parity-negative runs fail even if their confirmation economics pass.

Drift evidence should be task-conditioned, not only a parameter-norm proxy:

```bash
python -m randopt_lora_lab.logit_drift \
  --out results/logit_drift_dense \
  --data data/countdown_generated_1200_seed20260507.json \
  --perturbation-backend dense \
  --family dense_gaussian \
  --population 64 \
  --prompts 32 \
  --sigma-values 0.0005,0.001,0.002 \
  --max-mean-kl 0.05 \
  --min-top1-equal 0.95

python -m randopt_lora_lab.logit_drift \
  --out results/logit_drift_lora \
  --data data/countdown_generated_1200_seed20260507.json \
  --perturbation-backend lora \
  --family factor_gaussian_lora \
  --population 64 \
  --prompts 32 \
  --rank 8 \
  --sigma-values 0.0005,0.001,0.002 \
  --max-mean-kl 0.05 \
  --min-top1-equal 0.95

python -m randopt_lora_lab.drift_parity \
  --reference results/logit_drift_dense \
  --candidate results/logit_drift_lora \
  --out results/drift_parity_dense_vs_lora \
  --max-kl-ratio 1.1 \
  --max-logit-l2-ratio 1.1 \
  --min-top1-delta -0.01
```

The drift parity gate must use true nonnegative full-vocab next-token KL from
`logit_drift.py`; signed NLL deltas or anchor likelihood proxies are not valid
KL evidence.

First live drift gate:

```text
results/drift_parity_dense_vs_lora_rank8_p32_sigma001
reference: dense_gaussian, P=32, prompts=32, sigma=0.001
candidate: factor_gaussian_lora, rank=8, P=32, prompts=32, sigma=0.001
```

| metric | dense Gaussian | rank-8 factor LoRA | LoRA / dense |
| --- | ---: | ---: | ---: |
| KL(base || candidate) mean | 0.00162154 | 0.00139563 | 0.8607 |
| logit L2 mean | 185.597 | 176.894 | 0.9531 |
| top-1 equal mean | 1.0 | 1.0 | +0.0 |

Gate: pass. This closes the narrow drift-evidence gap for this sigma/rank
point. It does not prove quality parity, because the rank-sweep quality and
stability gates are still red.

## Requirement Checklist

| requirement | current evidence | status |
| --- | --- | --- |
| Dense Gaussian reference exists | `dense_gaussian` backend and P=16 baseline | partial |
| Serveable LoRA-style family exists | `factor_gaussian_lora` adapter path and vLLM LoRA probes | partial |
| Rank-r dense bridge exists | `projected_gaussian_rank_r` factors best rank-r dense projection | implemented, not yet run at scale |
| Quality parity | P=64 rank sweep failed at rank 8 and rank 32 | missing |
| Stability parity | parity report measures Spearman/top-k/regret for one panel; rank 8 Spearman -0.071, rank 32 Spearman -0.018 | missing multi-seed |
| Drift parity | `results/drift_parity_dense_vs_lora_rank8_p32_sigma001` compares true full-vocab next-token KL for dense Gaussian and rank-8 factor LoRA | passed at P=32 / 32 prompts / sigma 0.001 |
| Eval speed parity | P=64 HF reference path has LoRA slower than dense at rank 8 and rank 32; vLLM LoRA probes exist | partial, quality-coupled speed not proven |
| Convenience | LoRA adapters are materialized as portable safetensors | partial |
| Robustness | generated non-overlap data, cap-hit/malformed logging, paired holdout rows | partial |
| Eval validity | strict parser rescoring, semantic split audit, base row checks, cap/malformed thresholds | implemented, must pass per claim |
| Paper-aligned geometry | sparse SGD RLVR note added after arXiv 2602.07729 | hypothesis only |
| Sample-size aggregation | rank-32 top-4 score-weighted aggregate improved holdout, but with high cap-hit | promising, invalid until cap audit |
| Sparse-low-rank family | `sparse_low_rank_lora` implemented with density variants and variance matching | implemented, not run |
| Prompt robustness | prompt-relative report gate added; method claims require multiple valid prompt templates | implemented, not passed |
| vLLM backend parity | P=16 gate passed protocol/base/tensor checks but failed ranking with Spearman -0.181 | missing |
| Two-stage acceleration | tokenized vLLM proposal + PEFT confirmation recovered PEFT best at k=1 with 10.55x eval-only speedup and 1.93x speedup including vLLM load/build | systems path passed on one P64 panel |

## Next Gate

Run the result validity audit on every search run before comparing quality:

```bash
python -m randopt_lora_lab.result_validity \
  --run results/SEARCH_RUN \
  --out results/SEARCH_RUN/validity
```

Pass criteria:

```text
1. Base screen and base holdout rows are present.
2. Screen and holdout IDs are disjoint.
3. Screen and holdout semantic Countdown examples are disjoint.
4. Saved rows match the current strict parser.
5. Selected candidates do not exceed cap-hit or malformed thresholds.
6. Ensemble rows exist for every reported K.
```

This gate is deliberately separate from quality. A run can have high exact
score and still fail validity if it wins by truncation, stale parsing, repeated
examples, or missing per-prompt base rows.

First validity audit results:

```text
results/paper_style_p128_qwen3b/dense: failed only because the old summary
  lacked candidate_score_metric / ensemble_vote_metric metadata.
results/paper_style_p128_qwen3b/lora: same metadata failure.
results/vllm_lora_search_iso_s0p01_p512_stop: failed semantic split,
  stale-parser, cap-hit, malformed, and metadata checks.
```

That means the corrected local paper-style dense/LoRA runs are cleaner than the
older vLLM search rows, but they still are not current-valid claim artifacts
because their summaries predate the metric-metadata guard. Future reruns must
pass this audit before entering parity or goal-audit evidence.

Run the backend parity gate before any broader vLLM quality claim:

```bash
OUT_ROOT=results/backend_parity_gate \
FAMILY=factor_gaussian_lora \
POPULATION=64 \
PROMPTS=64 \
RANK=8 \
SIGMA=0.0075 \
scripts/run_backend_parity_gate.sh
```

Pass criteria:

```text
1. HF/PEFT and vLLM use the same candidate panel and protocol metadata.
2. Base screen and base holdout rows are saved for both backends.
3. Kept vLLM adapter tensors match the canonical materializer sample.
4. Saved PEFT/vLLM screen rows have zero exact disagreement, zero max
   candidate exact delta, no cap/malformed deltas, and high answer equality.
5. Spearman >= 0.85 and top-8 overlap >= 6 on the same screen split.
6. Selected-candidate regret versus the trusted HF/PEFT ranking is negligible.
```

Only after that gate passes should vLLM results be used as quality-selection
evidence. Until then, vLLM results are systems/plumbing evidence only.

There is a narrower systems claim that does not require vLLM-only selector
parity: vLLM can propose candidates and trusted PEFT/HF can confirm the
promoted set. The first tokenized P64 confirmation-economics gate passed:

```text
results/confirmation_economics_p64_tokenized_vllm
best recovered at k = 1
zero-regret k = 1
eval-only speedup = 10.55x
full-without-PEFT-load speedup = 1.93x
```

This should be treated as the current systems route. It is not a quality parity
result and does not rescue vLLM-only selection. Future runs should report both
gates separately:

```text
vLLM-only selector parity: strict, currently failing.
vLLM proposal + PEFT confirmation: recall/speed gate, currently passed on one P64 panel.
```

The current next entrypoint for a quality-coupled systems test is:

```bash
scripts/run_spectral_vllm_confirmation.sh
```

That script runs a dense PEFT reference, a matched factor-LoRA control, a
calibrated spectral-LoRA PEFT trusted arm, a tokenized multi-prompt vLLM
proposal arm for the same spectral family, dense-vs-spectral `parity_report`,
and same-family `confirmation_economics`. Read `docs/spectral_vllm_confirmation.md`
before interpreting the result. A confirmation pass is systems evidence only;
the spectral arm still needs PEFT dense-parity and validity evidence before any
quality claim.

First gate result: `results/backend_parity_gate_p16` failed. The run had
matching protocol metadata, saved base rows, and `576/576` sampled adapter tensor
checks passed across four kept adapters, but ranking parity failed:

```text
Spearman(PEFT, vLLM): -0.181164
top-8 overlap: 7/8
selected regret vs PEFT: 0.125
PEFT candidate/sec: 0.717830
vLLM candidate/sec: 6.616220
```

This preserves the systems speed signal but blocks vLLM as a selector of record.
The next diagnosis is not a larger vLLM search; it is base/zero/candidate
logit or next-token parity for the two disagreeing candidates.

Saved-output diff: `results/backend_parity_gate_p16/output_diff` shows the
ranking failure comes from real per-prompt output differences, not just aggregate
report formatting:

```text
common rows: 256
exact disagreement rate: 2.34375%
answer equal rate: 67.578125%
text equal rate: 55.078125%
max candidate exact delta: 0.125
```

The sparse exact reward makes a few row-level disagreements enough to invert a
P=16 ranking. That strengthens the case for a lower-level next-token/logit
parity probe before any more vLLM selection work.

The next probe command is:

```bash
python -m randopt_lora_lab.backend_next_token_probe \
  --out results/backend_next_token_probe_p16 \
  --data data/countdown_generated_1200_seed20260507.json \
  --prompts 8 \
  --seed 4242 \
  --rank 8 \
  --include-zero \
  --candidate factor_gaussian_lora:seed509771609:s0.0075:sign-1 \
  --candidate factor_gaussian_lora:seed1019282515:s0.0075:sign1
```

Expected gate: base and zero adapter should have near-identical next-token top-1
and top-k sets across PEFT and vLLM before candidate-level differences are
interpreted. If base/zero fails, the issue is backend decoding/logprob
semantics. If base/zero passes and candidates fail, the issue is LoRA adapter
application/scaling semantics.

First next-token probe: `results/backend_next_token_probe_p16` passed top-1
parity for base, zero, and the two disagreement candidates on 8 prompts
(`overall_top1_equal_rate=1.0`). That rules out a first-token top-1 mismatch as
the main reason for the P=16 ranking failure, but the common-token logprob
deltas were still large.

Short rollout probe:

```bash
python -m randopt_lora_lab.backend_rollout_probe \
  --out results/backend_rollout_probe_p16 \
  --data data/countdown_generated_1200_seed20260507.json \
  --prompts 8 \
  --seed 4242 \
  --rank 8 \
  --max-new-tokens 32 \
  --stop-at-answer \
  --include-zero \
  --candidate factor_gaussian_lora:seed509771609:s0.0075:sign-1 \
  --candidate factor_gaussian_lora:seed1019282515:s0.0075:sign1
```

Rollout result: `results/backend_rollout_probe_p16` shows base and zero are
mostly aligned (`text_equal_rate=0.875`), but adapter rollouts diverge even
though first-token top-1 matched. Candidate
`factor_gaussian_lora:seed509771609:s0.0075:sign-1` had `text_equal_rate=0.25`,
PEFT cap-hit `1.0`, vLLM cap-hit `0.0`, and mean absolute output-token delta
`16.625`. Candidate `factor_gaussian_lora:seed1019282515:s0.0075:sign1` had
`text_equal_rate=0.0`, PEFT cap-hit `0.875`, vLLM cap-hit `0.375`, and mean
absolute output-token delta `9.5`.

Interpretation: first-token checks are necessary but not sufficient. Exact
reward can hide backend disagreement when both completions are wrong; rollout
parity and cap/malformed parity must be part of the vLLM selector gate.

Then run a rank sweep before any broader LoRA-vs-dense claim:

```bash
BASE_OUT=results/gaussian_parity_rank_sweep \
RANKS=8,32 \
REUSE_DENSE=1 \
POPULATION=64 \
PROMPTS=64 \
HOLDOUT_PROMPTS=256 \
SIGMA=0.01 \
scripts/run_gaussian_parity_rank_sweep.sh
```

Pass criteria:

```text
1. factor_gaussian_lora and projected_gaussian_rank_r share the dense candidate panel.
2. factor_gaussian_lora has speed ratio >= 1.0 against dense.
3. factor_gaussian_lora selected-regret is <= 0 at matched dense screen score.
4. Spearman >= 0.85 and top-8 overlap >= 6 against dense.
5. The winning candidate does not increase cap-hit or malformed rate versus base.
6. The same conclusion holds at more than one seed.
```

If rank 8 fails and rank 32 passes, the project has a useful minimum-rank
finding, not a rank-8 parity result.

If projected rank-r passes but factor-Gaussian LoRA fails, the search family is
wrong even though the rank budget can carry useful directions.

If both projected and factor-Gaussian fail, low-rank serveable search is not yet
competitive with dense Gaussian on this task and model.

## Aggregate Probe Update

`results/aggregate_rank32_top4_score` tested whether a large candidate sample
can be turned into a stronger adapter by concatenating elite LoRA factors. This
is not dense-Gaussian parity; it tests a separate "large sample unlock" idea.

| base rank | aggregate rank | base holdout exact | aggregate holdout exact | holdout lift | holdout cap-hit | verdict |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 8 | 32 | 8.984% | 1.172% | -7.812 pp | 1.562% | fail |
| 32 | 128 | 8.984% | 14.062% | +5.078 pp | 25.781% | promising but not valid yet |

The rank-32 aggregate was the first positive evidence for combining sampled
perturbations after search, but the follow-up token-cap and prompt-robustness
audit failed. Under the default prompt the aggregate kept the same apparent
lift at caps 32/64/128, but cap-hit stayed around 25%. Under the reordered
semantically equivalent prompt, the base model remained protocol-valid while
the aggregate collapsed.

| prompt | protocol-valid caps | aggregate exact | base exact | lift | aggregate cap-hit | aggregate malformed | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| default | 3 | 14.062% | 8.984% | +5.078 pp | 25.4-25.8% | 1.953% | invalid: cap-hit regression |
| reordered | 3 | 0.391% | 5.859% | -5.469 pp | 53.1-82.0% | 98.047% | invalid: prompt collapse |
| xml | 0 | 6.250% | 5.469% | +0.781 pp | 76.9-85.5% | 20.3-21.1% | stress only: base malformed |

Prompt robustness report: `results/prompt_robustness_rank32_top4/report.md`.
Gate result: `pass=false`, `valid_prompt_variants=2`,
`passing_prompt_variants=0`, `min_lift_observed=-5.469 pp`.

Prompt robustness is now a required gate. A method is not allowed to claim
quality if it only works under one prompt template. Evaluation must report
candidate lift relative to the base model for each prompt template. Prompt
templates that collapse the base model's malformed/cap-hit rate are marked
protocol-invalid stress conditions, not used as evidence for or against the
method.

Selector-side scoring follows the same rule: collapsed base prompt variants are
excluded from candidate selection and holdout ranking, while still being logged
as stress diagnostics. This prevents a bad prompt rewrite from deciding the
winner, but it does not rescue a prompt-brittle method; a quality claim still
requires passing multiple base-valid prompt variants.

The prompt gate counts distinct protocol-valid prompt variants, not repeated
token caps. A variant passes only if every protocol-valid cap condition has
nonnegative lift and does not regress malformed or cap-hit rate beyond the
configured tolerance. Exact-score lift alone is not enough.

Prompt variants used for robustness must preserve the same answer contract:
tagged answer, one expression, every number exactly once, no equals sign, no
reasoning, and no extra text. Shorter prompts that omit these constraints have
already collapsed the base model and should be logged as stress tests rather
than used as robustness evidence.
