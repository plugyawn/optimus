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

## Requirement Checklist

| requirement | current evidence | status |
| --- | --- | --- |
| Dense Gaussian reference exists | `dense_gaussian` backend and P=16 baseline | partial |
| Serveable LoRA-style family exists | `factor_gaussian_lora` adapter path and vLLM LoRA probes | partial |
| Rank-r dense bridge exists | `projected_gaussian_rank_r` factors best rank-r dense projection | implemented, not yet run at scale |
| Quality parity | P=64 rank sweep failed at rank 8 and rank 32 | missing |
| Stability parity | parity report measures Spearman/top-k/regret for one panel; rank 8 Spearman -0.071, rank 32 Spearman -0.018 | missing multi-seed |
| Drift parity | update geometry audit reports update norm/effective rank/sparsity | missing logit drift / token KL matching |
| Eval speed parity | P=64 HF reference path has LoRA slower than dense at rank 8 and rank 32; vLLM LoRA probes exist | partial, quality-coupled speed not proven |
| Convenience | LoRA adapters are materialized as portable safetensors | partial |
| Robustness | generated non-overlap data, cap-hit/malformed logging, paired holdout rows | partial |
| Paper-aligned geometry | sparse SGD RLVR note added after arXiv 2602.07729 | hypothesis only |
| Sample-size aggregation | rank-32 top-4 score-weighted aggregate improved holdout, but with high cap-hit | promising, invalid until cap audit |
| Sparse-low-rank family | `sparse_low_rank_lora` implemented with density variants and variance matching | implemented, not run |
| Prompt robustness | prompt-relative report gate added; method claims require multiple valid prompt templates | implemented, not passed |

## Next Gate

Run a rank sweep before any broader claim:

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

The prompt gate counts distinct protocol-valid prompt variants, not repeated
token caps. A variant passes only if every protocol-valid cap condition has
nonnegative lift and does not regress malformed or cap-hit rate beyond the
configured tolerance. Exact-score lift alone is not enough.

Prompt variants used for robustness must preserve the same answer contract:
tagged answer, one expression, every number exactly once, no equals sign, no
reasoning, and no extra text. Shorter prompts that omit these constraints have
already collapsed the base model and should be logged as stress tests rather
than used as robustness evidence.
