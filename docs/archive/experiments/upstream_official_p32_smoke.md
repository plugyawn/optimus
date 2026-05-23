# Upstream Official Countdown P=32 Smoke

Date: 2026-05-07

This was a smoke test of the upstream RandOpt Countdown path after the local
8.20% dense/LoRA panel was found not to be an upstream-equivalent run.

## Setup

- upstream repo: `sunrainyg/RandOpt`
- upstream commit: `fbb774434f4e245fa7028797c40410aad2085908`
- model: `allenai/Olmo-3-7B-Instruct`
- data: official Countdown JSON linked by upstream
  (`VsonicV/es-fine-tuning-paper/countdown/data/countdown.json`)
- train samples: `200`
- test samples: `128`
- population: `32`
- sigmas: `0.0005,0.001,0.002`
- max tokens: `1024`
- top-K list induced by ratios: `3,1`
- hardware: `1x A100 80GB`

This is not an upstream-equivalent run because the full upstream population is `5000`
and the eval split here was shortened for a smoke test.

## Result

Upstream base reward means:

| Split | Reward |
| --- | ---: |
| train | 72.715% |
| test | 67.586% |

Screened population train rewards:

| Sigma | Count | Mean train reward |
| --- | ---: | ---: |
| 0.0005 | 9 | 75.459% |
| 0.001 | 10 | 75.312% |
| 0.002 | 13 | 71.600% |

Best screened candidates:

| Rank | Seed | Sigma | Train reward |
| ---: | ---: | ---: | ---: |
| 1 | 952224740 | 0.001 | 79.420% |
| 2 | 1688060228 | 0.0005 | 79.100% |
| 3 | 1102145665 | 0.0005 | 78.360% |

Ensemble numeric-vote accuracy on the 128-example smoke eval:

| K | Accuracy | Correct |
| ---: | ---: | ---: |
| 3 | 71.094% | 91/128 |
| 1 | 64.844% | 83/128 |

## Interpretation

This validates that the upstream setup can find positive train-reward
perturbations at small population and that the K=3 numeric-vote ensemble can
beat the upstream base test reward on this smoke panel. It does not validate the
8.20% local exact-only result, nor does it prove LoRA parity with dense Gaussian
RandOpt.

The earlier 8.20% result was a local Qwen/q-v-only/max-32-token/exact-answer
panel. It should be treated as a local stress test, not a reproduction of the
RandOpt paper.

Raw pulled artifacts are intentionally left under ignored `results/` paths:

- `results/upstream_randopt_official_p32/countdown_20260507_135251/results.json`
- `results/upstream_randopt_official_p32/countdown_20260507_135251/args.json`
- `results/upstream_randopt_official_p32/countdown_20260507_135251/model_saves/top_k_seeds.json`
- `results/logs/upstream_randopt_official_p32.log`
