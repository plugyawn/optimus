# QProj C2 Selector Calibration 2x P64

## Run

```text
date: 2026-05-08
out: results/qproj_c2_selector_calibration_2x_p64
runs:
  results/qproj_c2_vllm_shortlist_p64
  results/qproj_c2_vllm_shortlist_p64_seed20260508
select_k: 8
selectors:
  current vLLM selection score
  proposal exact
  default/reordered/xml exact
  prompt-robust lift reductions
  simple instability/malformed penalties
  ridge linear calibration from vLLM features
```

This is an offline gate for the q-only c2 accelerated cascade. It asks whether
the two P64 panels contain enough signal to calibrate vLLM-derived scores into a
selector that recovers the trusted dense PEFT screen winner on a held-out panel.

## Verdict

```text
gate: FAIL
fixed selector held-out pass count: 0/2
linear calibrated held-out pass count: 0/2
```

A pass required the selector chosen or trained on one panel to recover the dense
best within top-8 on the other panel. No tested selector did that.

## Per-Panel Evidence

For `results/qproj_c2_vllm_shortlist_p64`, every fixed selector missed the
dense best at top-8. The best positive Spearman among the simple selectors was
only `0.142`, and the current vLLM selection score had Spearman `0.003` with
the dense screen.

For `results/qproj_c2_vllm_shortlist_p64_seed20260508`, fixed selectors again
missed the dense best at top-8. Some selectors had zero dense regret at top-8
because they recovered another candidate with the same dense score, but the
actual dense-best seed/spec was still ranked outside the shortlist.

The held-out folds were worse than in-panel fitting:

| train | test | selector | Spearman | dense best rank | top-8 contains dense best | top-8 regret |
| --- | --- | --- | ---: | ---: | --- | ---: |
| seed20260508 | original | chosen fixed: default exact | 0.140 | 64 | false | 1.5625 pp |
| seed20260508 | original | linear calibrated | -0.357 | 28 | false | 6.2500 pp |
| original | seed20260508 | chosen fixed: valid min lift | 0.181 | 26 | false | 0.0000 pp |
| original | seed20260508 | linear calibrated | -0.255 | 28 | false | 1.5625 pp |

## Interpretation

This does not invalidate the operational cascade result. The two P64 panels
still show that `vLLM proposal + PEFT confirmation` can be faster than a full
dense PEFT screen while matching or beating dense strict holdout after
confirmation.

It does block the stronger claim that the current vLLM selector is a reliable
proxy for dense Gaussian RandOpt. The cheap scores do not rank dense winners
well enough, and a simple cross-panel calibration does not fix the problem.

The next high-leverage work should therefore focus on the perturbation family
and target-shape scaling, while keeping PEFT confirmation as the authority.
Do not scale the current selector without a new recall gate.
