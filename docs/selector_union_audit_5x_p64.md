# Selector Union Audit 5x P64

## Run

```text
date: 2026-05-08
out: results/selector_union_audit_5x_p64
runs:
  results/qproj_c2_vllm_shortlist_p64
  results/qproj_c2_vllm_shortlist_p64_seed20260508
  results/tscale_qv_p64_seed20260507
  results/tscale_qv_global_p64_seed20260507
  results/tscale_qkvo_global_p64_seed20260507
```

This is an offline selector audit over the saved P64 panels. It asks whether a
cheap vLLM-score policy would have sent the dense PEFT screen winner to PEFT
confirmation, or at least selected a candidate with negligible dense-screen
regret.

## Policies

The audit compares:

```text
current_selection: current robust selection_score
proposal_exact: vLLM candidate exact_mean
default_exact: default-prompt vLLM exact_mean
prompt_exact_rr: round-robin union of default/reordered/xml/proposal exact
prompt_lift_rr: round-robin union of prompt lifts
stability_rr: round-robin union of stability/lift heuristics
all_builtin_rr: round-robin union of all built-in selector columns
```

It also adds a runnable shortlist mode:

```bash
PYTHONPATH=. python -m randopt_lora_lab.selector_union_audit shortlist \
  --run results/tscale_qkvo_global_p64_seed20260507 \
  --out /tmp/shortlist.jsonl \
  --policy default_exact \
  --k 4
```

`scripts/run_vllm_shortlist_confirmation.sh` now accepts:

```text
SHORTLIST_POLICY=default_exact
```

When set, the script writes `shortlist_topK.jsonl` through
`selector_union_audit shortlist` instead of sorting by a single
`PROPOSAL_SCORE_COL`.

## Finding

Exact dense-best recall remains bad:

```text
No policy recovered the dense best on every run, even at k=16.
No policy recovered the dense best on every run at k=32 either.
```

The blocker is mostly the first q-only c2 panel:

```text
results/qproj_c2_vllm_shortlist_p64
  dense best rank by current_selection: 60/64
  dense best rank by default_exact:     64/64
  dense best rank by prompt_exact_rr:   64/64
```

That means exact dense-best recovery is not a realistic gate for the current
vLLM score surface.

## Practical Regret View

The more useful result is dense regret. `default_exact` top-4 has max
dense-screen regret of only one 64-prompt example across all five panels:

```text
policy: default_exact
k: 4
max dense-screen regret: 1/64 = 1.5625 percentage points
mean dense-screen regret: 1.25 percentage points
dense-best recall: 1/5
```

At k=8, no policy improves the max regret over this:

| policy | k | dense-best recall | mean regret | max regret |
| --- | ---: | ---: | ---: | ---: |
| default_exact | 4 | 1/5 | 0.0125 | 0.015625 |
| default_exact | 8 | 1/5 | 0.009375 | 0.015625 |
| prompt_exact_rr | 16 | 3/5 | 0.003125 | 0.015625 |
| prompt_lift_rr | 16 | 3/5 | 0.003125 | 0.015625 |
| stability_rr | 16 | 3/5 | 0.003125 | 0.015625 |

So the next selector claim should not be "recovers dense best." The realistic
claim to test live is:

```text
default_exact top-4 or prompt_exact_rr top-16 can keep dense-screen regret under
one screen example while preserving end-to-end speed.
```

## Next GPU Test

Use q-only c2 because it has the strongest existing operational evidence, but
the next primary GPU test should create a fresh corrected vLLM screen with
base-healthy prompt variants:

```bash
MODE=confirm scripts/run_qproj_c2_corrected_confirmation.sh
```

The older existing-panel replay remains useful for provenance forensics because
it preserves the saved activation basis from `vllm/family_state.pt`, but it
inherits the old `default,reordered,xml` source summary and should not be used
as the primary prompt-agnostic quality claim:

```bash
SOURCE_ROOT=results/qproj_c2_vllm_shortlist_p64 \
OUT_ROOT=results/qproj_c2_vllm_shortlist_p64_default_exact_k4 \
FAMILY=activation_spectral_lora_c2 \
TARGETS=q_proj \
SEED=20260507 \
DATA=data/countdown_generated_1200_seed20260507.json \
POPULATION=64 PROMPTS=64 HOLDOUT_PROMPTS=128 VLLM_HOLDOUT_PROMPTS=8 \
SHORTLIST_K=4 RANK=32 SIGMA_VALUES=0.0005,0.001,0.002 \
ENSEMBLE_KS=1,4 MAX_NEW_TOKENS=128 HF_BATCH_SIZE=16 \
VLLM_PROMPT_INPUT=token_ids \
VLLM_PROMPT_VARIANTS=default,reordered \
VLLM_SCORE_MODE=robust_mean \
SHORTLIST_POLICY=default_exact \
scripts/run_existing_vllm_shortlist_confirmation.sh
```

For a higher-recall but more expensive test:

```text
SHORTLIST_POLICY=prompt_exact_rr
SHORTLIST_K=16
CONFIRM_KS=1,2,4,8,16
CONFIRM_MAX_K=16
```

Success should be judged by PEFT-confirmed strict holdout, validity, and full
wall-clock speedup. Dense-best recall is diagnostic only.

## Verdict

```text
current robust selector: failed
cross-panel calibrated selector: failed
union dense-best recall: failed
low-regret default_exact top-4: promising but unconfirmed
```

The immediate next step is a live PEFT confirmation of the low-regret shortlist
policy, not another perturbation family.
