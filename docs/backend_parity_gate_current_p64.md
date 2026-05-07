# Backend Parity Gate P64

Run: `results/backend_parity_gate_current_p64`

Date: 2026-05-07

Model: `Qwen/Qwen2.5-3B-Instruct`

Family: `factor_gaussian_lora`, rank 8, targets `q_proj,v_proj`, sigma `0.0075`, antithetic P=64.

Protocol: 64 unique screen prompts, 8 unique holdout prompts, `max_new_tokens=32`, `stop_at_answer=true`, `promote=0`.

## Result

The gate failed.

This is the useful failure mode: adapter tensor parity passed, but backend generation/ranking/output parity did not.

| Metric | PEFT | vLLM |
| --- | ---: | ---: |
| Eval candidate/sec | 0.1588 | 3.5452 |
| Eval speedup | 1.0x | 22.3x |
| vLLM full-run candidate/sec, including load/build | | 0.6062 |
| vLLM full-run speedup, including load/build | | 3.8x |
| Base screen exact | 0.0625 | 0.046875 |
| Base holdout exact | 0.0 | 0.0 |

## Gate Checks

| Check | Result |
| --- | ---: |
| Protocol metadata | pass |
| Base rows present | pass |
| Adapter tensor parity | pass, 1152 tensors checked |
| Ranking correlation | fail |
| Output diff parity | fail |

Ranking details:

| Metric | Value |
| --- | ---: |
| Spearman | 0.6191 |
| Spearman gate | 0.85 |
| Pearson | 0.7931 |
| Top-4 overlap | 4/4 |
| Top-8 overlap | 6/8 |
| Top-16 overlap | 10/16 |
| Trusted best score | 0.21875 |
| vLLM best trusted score | 0.21875 |
| Selected regret vs PEFT trusted best | 0.0 |
| Mean abs score delta | 0.01953 |
| Max abs score delta | 0.125 |

Output-diff details:

| Metric | Value |
| --- | ---: |
| Exact disagreement rate | 0.02588 |
| Answer equal rate | 0.59790 |
| Text equal rate | 0.52661 |
| Max abs exact delta by candidate | 0.125 |
| Mean abs exact delta by candidate | 0.01953 |
| Max abs cap-hit delta by candidate | 0.953125 |
| Mean abs cap-hit delta by candidate | 0.11841 |
| Max abs malformed delta by candidate | 0.703125 |
| Mean abs malformed delta by candidate | 0.08838 |

Validity audit:

The uniqueness, semantic-disjointness, base-row, exact-score-metric, and strict parser checks passed for both backends. The validity audit fails `candidate_holdout_rows_present` because this was an intentional `promote=0` backend gate, so no candidate holdout rows were generated.

## Interpretation

The old concern that PEFT and vLLM were materializing different LoRA tensors is fixed for this panel. The fast path generated adapters that exactly match the canonical tensor generator.

vLLM is still not trustworthy as the sole winner-selection backend. It is much faster eval-only and it picked the same top candidate in this run, but the score surface differs enough that the strict gate should continue to reject it: Spearman is below threshold, exact output disagreement is nonzero, and cap/malformed deltas are large.

The immediate operating rule is:

- use vLLM for fast exploratory screens only;
- confirm top-K candidates through PEFT/HF before making quality claims;
- do not scale a vLLM-only quality run until output/ranking parity is fixed or the acceptance policy explicitly tolerates backend drift.

Next highest-leverage checks:

1. Add a small base/candidate logits parity probe before generation to separate model math drift from decoding/stop behavior drift.
2. Run a deterministic generation micro-panel with identical tokenizer settings and stop criteria, then compare base-only PEFT vs vLLM outputs.
3. Keep the fast vLLM screen, but make PEFT confirmation of top 8 or top 16 part of the search pipeline.
4. Run quality claims only on a promoted final holdout with candidate rows; this P64 gate is not a quality-validation run.
