# Power Subspace Audit

Date: 2026-05-08

## Question

Can the accumulated perturbation archive be used to make search cheaper by
prefiltering candidate seeds before any GPU rollout?

The intended use is not per-candidate randomized SVD. That is too expensive for
the hot loop. The useful form is amortized:

```text
learn a low-dimensional direction sketch from old scored candidates
sample a very large cheap seed pool
keep only high-scoring proposed candidates
evaluate that candidate file with vLLM/SGLang
confirm top-K with trusted PEFT/HF
```

## Implementation

Added:

- `randopt_lora_lab.subspace_audit`
- `randopt_lora_lab.subspace_propose`
- `--candidate-file` support in `randopt_lora_lab.vllm_lora_search`

The audit uses deterministic candidate sketches. The corrected sketch treats
`seed` as the direction and `sigma` as scale; unit-scale audits remove the
sigma confound. It reports:

- learned mean-direction predictor
- randomized power-energy subspace predictor
- ridge predictor
- shuffled/permutation controls
- antithetic finite-difference sign test
- row-random and source-heldout splits

## Strongest Offline Result

Prior rows:

```text
results/phase8_extra_pod1/search_chunk8_p1024
results/phase8_sustain_pod1/search_chunk4_p1024
results/phase8_systems_pod1/search_chunk16_p512
results/search_iso_s0p01_p512_seed5678
results/search_iso_s0p0075_p512_seed2468
results/search_iso_s0p0125_p512_seed2468
```

Strict filter:

```text
cap_hit_mean <= 0.05
malformed_mean <= 0.05
family = isotropic
rows = 1862
antithetic pairs = 425
split = source-heldout
feature_scale = unit
```

Best corrected audit:

```text
results/subspace_audit_iso_source_strict_unit_direction_d512_k16
```

| method | Spearman | top-16 overlap | top-16 mean lift | selected-best regret |
| --- | ---: | ---: | ---: | ---: |
| power_energy_k16 | 0.337 | 4.73 | +0.0447 | 0.0078 |
| perm_power_energy_k16 | 0.095 | 0.13 | +0.0053 | 0.0734 |
| mean_direction | 0.271 | 3.43 | +0.0471 | 0.0146 |
| perm_mean_direction | -0.014 | 0.30 | +0.0049 | 0.0641 |

Antithetic sign test:

| method | pair Spearman | sign accuracy | chosen lift |
| --- | ---: | ---: | ---: |
| antithetic_mean_gradient | 0.324 | 0.615 | +0.0051 |
| perm_antithetic_mean_gradient | 0.010 | 0.510 | +0.0004 |

This is real enough to test as a proposal prior. It is not enough to claim final
quality, because the labels are screen exact scores from prior runs rather than
new heldout PEFT-confirmed quality.

## Generated Proposal

The clean candidate file is:

```text
results/subspace_power_unit_proposal_iso_s0p01_pool100k_keep512/candidates.jsonl
```

It was generated from:

```text
pool = 100000
keep = 512
family = isotropic
sigma = 0.01
antithetic = true
feature_scale = unit
sketch_dim = 512
components = 16
score_mode = power_energy
```

This is intentionally fixed-sigma. A multi-sigma proposal should be treated as a
separate scale ablation.

## Next GPU Test

Run two matched vLLM screens, then PEFT-confirm top-K:

```bash
PYTHONPATH=. python -m randopt_lora_lab.vllm_lora_search \
  --out results/vllm_subspace_power_unit_s0p01_p512 \
  --candidate-file results/subspace_power_unit_proposal_iso_s0p01_pool100k_keep512/candidates.jsonl \
  --family isotropic \
  --rank 32 \
  --prompts 64 \
  --holdout-prompts 256 \
  --promote 16 \
  --max-new-tokens 128 \
  --stop-at-answer \
  --prompt-input token_ids \
  --chunk-adapters 8 \
  --max-loras 8 \
  --ensemble-ks 1,4,8,16
```

Matched control:

```bash
PYTHONPATH=. python -m randopt_lora_lab.vllm_lora_search \
  --out results/vllm_random_iso_s0p01_p512_matched \
  --family isotropic \
  --population 512 \
  --sigma 0.01 \
  --antithetic \
  --rank 32 \
  --prompts 64 \
  --holdout-prompts 256 \
  --promote 16 \
  --max-new-tokens 128 \
  --stop-at-answer \
  --prompt-input token_ids \
  --chunk-adapters 8 \
  --max-loras 8 \
  --ensemble-ks 1,4,8,16
```

Pass condition:

```text
subspace proposal must beat matched random on confirmed top-K recall,
heldout exact / ensemble exact, and selected regret without worse cap/malformed.
```

If it passes, the large-sample search trick is:

```text
sample 100k-1M cheap candidate sketches
evaluate only the best 512-2048 adapters
confirm top-K with trusted backend
```

That is a search-speed improvement without requiring a custom LoRA kernel.

If it fails, power iteration remains an offline diagnostic only.
