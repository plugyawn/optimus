# vLLM Prompt-Robust Selection Smoke

Date: 2026-05-07

Pod: `707ea4d6223044518e792d483c7bb421`, 1x A100 80GB, terminated after artifact pull.

Command shape:

```bash
python3 -m randopt_lora_lab.vllm_lora_search \
  --family sparse_low_rank_lora_d0p25 \
  --population 16 \
  --promote 4 \
  --rank 8 \
  --sigma 0.01 \
  --prompts 16 \
  --holdout-prompts 32 \
  --prompt-variants default,reordered \
  --max-new-tokens 64 \
  --stop-at-answer
```

The paired runs used the same candidate panel and differed only in selection
mode: `robust_min` versus `exact`.

## Throughput

| run | score mode | screen candidate/sec | end-to-end candidate/sec | eval elapsed | load |
| --- | --- | ---: | ---: | ---: | ---: |
| `vllm_robust_smoke_p16_fixed` | `robust_min` | 2.413 | 1.932 | 8.280s | 53.518s |
| `vllm_exact_smoke_p16_fixed` | `exact` | 3.610 | 2.516 | 6.360s | 33.467s |

The smoke includes two prompt variants, so the robust scorer is evaluating twice
as many prompt-candidate pairs as a single-prompt screen. The observed overhead
is less than 2x because vLLM batches mixed LoRA requests.

## Selection Difference

| score mode | promoted seeds | screen notes |
| --- | --- | --- |
| `robust_min` | `2121586239`, `685255715`, `1151155068`, `2103139804` | avoided high cap-hit candidate `1688041804` and high malformed candidate `1982656189` |
| `exact` | `2121586239`, `1151155068`, `1688041804`, `1982656189` | promoted candidates with max malformed regression up to 0.750 and max cap-hit regression up to 0.5625 |

On holdout, the exact-selected brittle candidates remained bad:

| seed | exact holdout exact | holdout malformed | holdout cap-hit | min condition score |
| ---: | ---: | ---: | ---: | ---: |
| `1688041804` | 4.688% | 48.438% | 31.250% | -1.375 |
| `1982656189` | 4.688% | 29.688% | 3.125% | -0.625 |

The robust-selected candidate `685255715` was the cleanest promoted candidate
on this tiny smoke:

| seed | robust holdout exact | holdout malformed | holdout cap-hit | min condition score |
| ---: | ---: | ---: | ---: | ---: |
| `685255715` | 10.938% | 1.562% | 0.000% | 0.0625 |

## Verdict

This is not a quality claim; it is a plumbing and selection-behavior smoke. It
shows that the accelerated vLLM path can evaluate a prompt ensemble, compute
base-relative malformed/cap-hit penalties, and avoid prompt-brittle exact-score
winners on the same candidate panel.
