# vLLM Sparse d0p125 Shortlist P32 Probe

Run:

```text
results/vllm_shortlist_sparse_d0p125_p32_probe
```

Command shape:

```bash
OUT_ROOT=results/vllm_shortlist_sparse_d0p125_p32_probe \
FAMILY=sparse_low_rank_lora_d0p125 \
POPULATION=32 \
SHORTLIST_K=8 \
PROMPTS=64 \
HOLDOUT_PROMPTS=128 \
VLLM_HOLDOUT_PROMPTS=8 \
RANK=32 \
SIGMA_VALUES=0.0005,0.001,0.002 \
MAX_NEW_TOKENS=128 \
HF_BATCH_SIZE=16 \
VLLM_MAX_LORAS=16 \
VLLM_CHUNK_ADAPTERS=16 \
scripts/run_vllm_shortlist_confirmation.sh
```

## Verdict

```text
vLLM systems path: pass
sparse d0p125 as dense-RandOpt replacement: fail
shortlist quality gate: fail
rollout validity: mostly pass, with corrected token-accounting caveat
```

The accelerated path is real: vLLM screened all 32 sparse LoRA candidates in
`26.05s`, about `1.23s/candidate`, with `235.8 prompts/s` on the A100. But the
proposal ranking did not recover the dense Gaussian reference winner, even after
PEFT-confirming the top 8 shortlist entries.

## Dense Reference

Dense reference was the authority for the quality gate.

| metric | value |
| --- | ---: |
| dense screen population | 32 |
| dense screen time | 1001.99s |
| base screen exact | 3.125% |
| base holdout exact | 7.031% |
| dense best screen exact | 9.375% |
| dense best screen candidate | `dense_gaussian:seed2071213761:s0.002:sign1` |
| dense validity audit | pass |

The dense screen had a screen tie at 9.375%; the dense-confirmation comparator
selected `seed2071213761:s0.002` as the dense best for the regret table.

## Shortlist Confirmation

| k | confirmed pick | confirmed screen | dense score at pick | dense regret | dense best recovered | eval-only speedup | full speedup excl. dense load |
| ---: | --- | ---: | ---: | ---: | --- | ---: | ---: |
| 1 | `sparse_low_rank_lora_d0p125:seed271202801:s0.002:sign1` | 4.6875% | 4.6875% | 4.6875 pp | no | 27.41x | 7.69x |
| 2 | `sparse_low_rank_lora_d0p125:seed2020458541:s0.001:sign1` | 7.8125% | 4.6875% | 4.6875 pp | no | 16.63x | 6.51x |
| 4 | `sparse_low_rank_lora_d0p125:seed2020458541:s0.001:sign1` | 7.8125% | 4.6875% | 4.6875 pp | no | 8.47x | 4.72x |
| 8 | `sparse_low_rank_lora_d0p125:seed2103302888:s0.0005:sign1` | 7.8125% | 3.125% | 6.25 pp | no | 4.38x | 3.11x |

Summary:

```text
zero dense-regret k: none
dense best recovered k: none
```

This means the current sparse d0p125 proposal is a fast selector, but not a good
dense-RandOpt selector.

## Rollout Probe

Manual rollout inspection showed why a 3B model is scoring around 3-10% on this
Countdown setup:

```text
The outputs are usually clean single equations in <answer>...</answer>.
The failures are mostly wrong arithmetic, not extraction failures.
The prompt is answer-only, not paper-style chain-of-thought.
```

Representative base failures were clean but wrong, e.g. for target 61 with
numbers `[90, 12, 4, 37]`, the model emitted:

```text
<answer>90-12-37+4</answer>
```

This is parseable and uses valid syntax, but evaluates to 45, not 61.

## Corrected Cap/Prompt Diagnostic

Follow-up run:

```text
results/cap_stability_sparse_shortlist_corrected_p4
```

This reran the top confirmed sparse shortlist candidates and a score-weighted
aggregate under caps `32,64,128` and prompt variants `default,reordered`, with
corrected raw-vs-visible token accounting.

Main findings:

```text
1. Scores were stable across caps 32/64/128.
2. cap_hit remained 0.0 in this diagnostic.
3. Prompt wording moved scores materially and inconsistently across split.
4. Aggregate did not improve over base.
```

Selected rows:

| split | prompt | cap | base | best elite | aggregate |
| --- | --- | ---: | ---: | ---: | ---: |
| screen | default | 32/64/128 | 3.125% | 6.25% | 3.125% |
| screen | reordered | 32/64/128 | 9.375% | 9.375% | 6.25% |
| holdout | default | 32/64/128 | 6.25% | 10.9375% | 6.25% |
| holdout | reordered | 32/64/128 | 3.125% | 7.8125% | 4.6875% |

The best elite varies by prompt/split. This is not a prompt-agnostic quality
claim.

## Token Accounting Caveat

The run exposed a harness issue in the HF generation path. The custom stopping
criterion stops only when every row in the batch has emitted `</answer>`. Rows
that answer early keep generating hidden continuation until the slowest row
finishes or the cap is reached. The decoded text is truncated at `</answer>`, so
semantic scoring remains based on the visible answer, but older `output_tokens`
were not a reliable measure of visible answer length.

Patch added after the run:

```text
visible output_tokens: tokens through first </answer>
raw_output_tokens: full generated tokens before truncation
hidden_after_answer_tokens: raw - visible
cap_hit: based on raw generated tokens
```

The corrected diagnostic shows the distinction clearly. Example: aggregate
holdout, reordered prompt, cap 128:

```text
visible output_tokens: 6577
raw_output_tokens: 8128
cap_hit_mean: 0.0
exact_mean: 4.6875%
```

So the quality scores are not cap-sensitive in this diagnostic, but speed and
token accounting from older HF runs should be interpreted with this caveat.

## Decision

Do not scale sparse d0p125 as currently configured.

What survives:

```text
1. vLLM multi-LoRA screening is fast enough to matter.
2. Sparse d0p125 can occasionally find prompt-local improvements.
3. The current proposal ranking is not aligned enough with dense Gaussian RandOpt.
```

Next high-leverage move:

```text
Use the A100 for proposal-quality diagnostics, not larger population search:
  - compute ranking correlations between vLLM sparse scores, PEFT sparse scores, and dense reference scores;
  - test a same-seed structured family that is explicitly fitted to match dense q/v perturbation effects;
  - keep dense reference as the authority and require zero dense regret or dense-best recovery at small k.
```

## Stable Tie-Break Recheck

After the first report, shortlist tie-breaking was patched to preserve original
adapter order on score ties instead of reverse lexicographic candidate order.
This matters because the vLLM scores are coarse and many candidates tie.

Recomputed shortlists from the same vLLM artifacts:

```text
stable selection top candidate: sparse_low_rank_lora_d0p125:seed1219141227:s0.002:sign1
exact-mean top candidate: sparse_low_rank_lora_d0p125:seed414161978:s0.001:sign1
```

The corrected stable-selection top-8 included dense-good seed specs earlier, so
we reran PEFT confirmation only for that shortlist:

```text
results/vllm_shortlist_sparse_d0p125_p32_probe/confirmed_stable_selection
results/vllm_shortlist_sparse_d0p125_p32_probe/shortlist_dense_confirmation_stable_selection
```

Result:

```text
zero dense-regret k: none
dense best recovered k: none
```

Stable confirmation table:

| k | confirmed pick | confirmed screen | dense score at pick | dense regret | speedup excl. dense load |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `sparse_low_rank_lora_d0p125:seed414161978:s0.001:sign1` | 6.25% | 4.6875% | 4.6875 pp | 6.75x |
| 2 | `sparse_low_rank_lora_d0p125:seed414161978:s0.001:sign1` | 6.25% | 4.6875% | 4.6875 pp | 6.31x |
| 4 | `sparse_low_rank_lora_d0p125:seed2020458541:s0.001:sign1` | 7.8125% | 4.6875% | 4.6875 pp | 4.73x |
| 8 | `sparse_low_rank_lora_d0p125:seed2020458541:s0.001:sign1` | 7.8125% | 4.6875% | 4.6875 pp | 3.72x |

This sharpens the interpretation. The earlier shortlist ordering was partly a
tie-break artifact, but correcting it does not rescue sparse d0p125. Even when a
dense-good seed/sigma appears early, the sparse-LoRA perturbation with that same
seed/sigma is not the same perturbation and does not preserve dense behavior.

The remaining failure is structural:

```text
same seed/sigma in sparse LoRA != dense Gaussian perturbation
vLLM sparse scores are tie-heavy and weakly correlated with dense scores
PEFT confirmation selects sparse-family winners, not dense-family winners
```
