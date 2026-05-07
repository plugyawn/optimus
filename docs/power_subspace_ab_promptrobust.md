# Power Subspace Proposal A/B

Date: 2026-05-08

## Setup

This run tested whether the offline power-subspace signal translated into a faster or higher-quality vLLM LoRA search proposal.

- Model: `Qwen/Qwen2.5-3B-Instruct`
- Backend: vLLM `0.20.1`, single-process V1 disabled (`VLLM_ENABLE_V1_MULTIPROCESSING=0`)
- GPU: 1x A100 80GB
- Data: `data/countdown_generated_1200_seed20260507.json`
- Screen / holdout: 64 / 256 Countdown examples
- Population: 512 candidates per arm
- LoRA: rank 32, `q_proj,v_proj`
- Sigma: `0.01`
- Prompt variants: `default,reordered`
- Selector: `robust_mean`
- Generation: `max_new_tokens=32`, `stop_at_answer`
- Ensembles: top-k = 1, 4, 8, 16

Important caveat: this is a generated local Countdown stress set, not the official upstream paper dataset. It is an internal A/B for proposal quality, not an official paper reproduction.

## Result

| Arm | Base holdout exact | Top-1 holdout exact | Ensemble k=1 | Ensemble k=4 | Ensemble k=8 | Ensemble k=16 | Candidates/sec | Screen candidates/sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| matched random isotropic | 9.18% | 12.89% | 11.33% | 15.04% | 16.41% | 18.95% | 1.587 | 1.771 |
| power-subspace proposal | 9.18% | 13.09% | 12.70% | 15.23% | 17.97% | 19.14% | 1.556 | 1.735 |

Delta, power-subspace minus matched random:

- Best ensemble holdout exact: `+0.001953125` absolute, or `+0.195` percentage points.
- Best top-candidate holdout exact: `+0.001953125` absolute, or `+0.195` percentage points.
- Candidate throughput: `-0.0304` candidates/sec.
- Screen throughput: `-0.0362` candidates/sec.

## Interpretation

The practical result is a tie. Power-subspace selection had a real offline audit signal, but in the actual vLLM P=512 search screen it did not materially beat matched random isotropic sampling. The observed +1/512 holdout difference is too small to justify scaling this proposal rule.

Both arms improved over the base model through selection and ensembling, but that is evidence for random LoRA search plus top-k voting, not evidence that the current power-subspace proposal is better.

Prompt robustness was acceptable for this run, but not decisive:

- Random k=16: default 19.14%, reordered 18.75%.
- Power-subspace k=16: default 19.92%, reordered 18.36%.

The follow-up proposal audit sharpened the failure mode:

| Diagnostic | Power-subspace | Matched random |
| --- | ---: | ---: |
| prompt-variant selection Spearman | 0.206 | 0.236 |
| prompt-variant exact Spearman | 0.139 | 0.184 |
| top-16 prompt-variant overlap | 1/16 | 0/16 |
| screen selection vs holdout exact Spearman on promoted candidates | 0.298 | 0.534 |
| screen top-16 exact mean | 12.01% | 12.50% |
| screen valid fraction | 9.18% | 8.20% |

So the issue is not only that final holdout tied. The proposal did not improve the actual screen pool, and the screen ranking remained very prompt-sensitive. Its small holdout edge came from one promoted candidate and is not strong enough to distinguish from random search noise.

The current proposal rule ranks candidates by sign-symmetric `power_energy`. That may be throwing away useful sign and coefficient information. If power iteration remains useful, it should next be tested as a small learned coefficient/sign search inside the subspace, not as a simple top-energy candidate filter.

## Decision

Do not scale this configuration.

Keep power/subspace tools as diagnostics and as a possible basis for a more direct coefficient search, but the next quality claim must beat matched random under the same prompt-robust screen and should PEFT-confirm top candidates before being treated as real.

## Artifact Index

- A/B summary: `results/subspace_ab_power_promptrobust_gen1200/summary.json`
- A/B log: `results/subspace_ab_power_promptrobust_gen1200/run.log`
- Proposal audit: `results/proposal_audit_subspace_power_vs_random_gen1200/`
- Power-subspace arm: `results/vllm_subspace_power_unit_s0p01_p512_promptrobust_gen1200/`
- Matched-random arm: `results/vllm_random_iso_s0p01_p512_promptrobust_gen1200/`
