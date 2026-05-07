# Confirmation Economics P64

Date: 2026-05-07

Purpose: estimate the practical fallback path after vLLM selector parity failed:

```text
vLLM proposes top-K candidates -> PEFT/HF confirms only those K -> select by PEFT/HF score
```

This does not make vLLM a trusted selector. It tests whether vLLM can be a fast proposal backend while PEFT/HF remains the selector of record.

Artifacts:

- `results/confirmation_economics_p64_tokenized_vllm`
- `results/confirmation_economics_p64_string_vllm`

Reference trusted run:

- `results/backend_parity_gate_p64_tokenized_vllm/peft`
- P=64, 64 screen prompts, PEFT/HF full screen time estimate: `403.04s`
- PEFT/HF best candidate: `factor_gaussian_lora:seed1411240924:s0.0075:sign1`
- PEFT/HF best score: `0.21875`

## Token-ID vLLM Proposal

Token-ID vLLM proposal screen:

- proposal screen time: `31.50s`
- vLLM load + adapter build: `170.48s`
- PEFT winner appears at proposal rank 1

| Confirm K | Recovers PEFT best | PEFT confirm sec | Proposal+confirm sec | Eval-only speedup | Full without PEFT-load speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | true | 6.70 | 38.21 | 10.55x | 1.93x |
| 2 | true | 10.55 | 42.06 | 9.58x | 1.90x |
| 4 | true | 24.07 | 55.57 | 7.25x | 1.78x |
| 8 | true | 50.80 | 82.31 | 4.90x | 1.59x |
| 16 | true | 98.75 | 130.25 | 3.09x | 1.34x |
| 32 | true | 193.23 | 224.74 | 1.79x | 1.02x |

## String-Prompt vLLM Proposal

String-prompt vLLM was faster but had worse base-score contract behavior:

- proposal screen time: `18.05s`
- vLLM load + adapter build: `87.52s`
- PEFT winner appears at proposal rank 1

| Confirm K | Recovers PEFT best | PEFT confirm sec | Proposal+confirm sec | Eval-only speedup | Full without PEFT-load speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | true | 6.70 | 24.76 | 16.28x | 3.59x |
| 2 | true | 13.39 | 31.44 | 12.82x | 3.39x |
| 4 | true | 24.07 | 42.12 | 9.57x | 3.11x |
| 8 | true | 48.59 | 66.64 | 6.05x | 2.61x |
| 16 | true | 98.92 | 116.97 | 3.45x | 1.97x |
| 32 | true | 196.62 | 214.67 | 1.88x | 1.33x |

## Interpretation

For this seed, the viable end-to-end systems story is:

```text
Token-ID vLLM proposal + PEFT/HF confirmation
```

At K=4, it recovers the PEFT/HF winner and gives an estimated `7.25x` eval-only speedup over evaluating all 64 candidates in PEFT/HF. At K=8, it still recovers the winner with `4.90x` eval-only speedup.

The full-run speedup including vLLM load/build is much smaller on P=64 because cold-start dominates. That should improve with larger populations or persistent workers, but it is not proven by this P64 run.

## Caveats

This is not a quality-parity proof:

- it is one P64 panel;
- the vLLM top-1 happened to be the PEFT top-1;
- selector parity still failed by Spearman/output-diff gates;
- PEFT model-load time is not included in the full-without-PEFT-load estimate;
- no heldout candidate quality claim is made because this backend gate used `promote=0`.

The next robustness check should repeat confirmation economics across multiple seeds and larger populations, reporting:

- probability the PEFT top-1 appears in vLLM top-K;
- PEFT regret after confirmation;
- eval-only and full-run speedup;
- heldout result for PEFT-confirmed winners.

Until that passes, the method is a promising proposal engine, not a completed replacement for full PEFT/HF RandOpt search.
