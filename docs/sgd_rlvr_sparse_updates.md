# SGD RLVR Sparse-Update Note

Source: arXiv 2602.07729, "Do We Need Adam? Surprisingly Strong and Sparse Reinforcement Learning with SGD in LLMs".

## Read

The paper argues that RLVR optimization behaves differently from supervised next-token training. In their experiments across math, code, and RL with evolving verifiable environments, plain SGD matches or often beats AdamW while using much less optimizer state.

The key result for this project is not only optimizer memory. It is update geometry:

```text
SGD RLVR updates are extremely sparse.
SGD RLVR updates have much lower effective rank than AdamW updates.
Momentum and adaptive learning rates are not consistently helpful in RLVR.
```

Their Table 4 reports SGD-style GRPO updates with roughly 99.84%-99.99% sparsity, while AdamW/RMSProp are much denser. The reported 99%-energy effective ranks for SGD updates are around the mid-20s in Qwen runs, while AdamW/RMSProp are much higher, often near the high-80s.

## Implication For RandOpt

This weakens dense Gaussian as the *natural* gold-standard perturbation family.

Dense Gaussian is still a useful baseline because it is simple and hard to game. But if RLVR's actual learned update is sparse and low-effective-rank, then "full dense Gaussian parity" is not the final scientific target. The better target is:

```text
Can a serveable perturbation family match or beat dense Gaussian search
while also matching the sparse/low-rank geometry of successful RLVR updates?
```

That makes LoRA more plausible as a systems-friendly search family, but it does not prove rank-8 LoRA is sufficient. The paper's reported effective ranks are closer to rank-24/26 than rank-8 in several Qwen settings, and the updates are sparse in coordinates as well as low-rank in matrices.

## New Hypotheses

1. `factor_gaussian_lora` is too smooth. It matches dense Gaussian expected Frobenius norm, but it is not sparse and not necessarily aligned with SGD RLVR update spectra.
2. `projected_gaussian_rank_r` is a cleaner bridge baseline than dense Gaussian alone, especially at ranks 8, 16, 32, and 64.
3. A better family may be `sparse_low_rank`: per-module low-rank factors plus coordinate or block sparsification after materialization.
4. The rank should be swept as a geometry parameter, not fixed at 8. Rank 32 is now a particularly important test point.
5. Momentum-style accumulation across random perturbation winners should be treated skeptically unless it passes a probe audit; the paper's momentum analysis suggests stale directions can be unhelpful in non-stationary RL.

## Geometry Audit To Add

For every perturbation family and every accepted update, report:

```text
l0_sparsity_threshold_1e-5
per-matrix 99%-energy effective rank
layerwise sparsity distribution
update Frobenius norm
logit L2 drift
token-distribution KL or approximation
exact reward lift
```

A family should be considered more RLVR-like if it improves search utility while moving toward the paper's sparse/low-rank update profile without increasing malformed, cap-hit, or drift failures.

Run the current standalone geometry audit with:

```bash
python -m randopt_lora_lab.update_geometry \
  --out results/report/update_geometry \
  --rank 8 \
  --sigma 0.01
```

## Cautions

The paper does not prove LoRA RandOpt parity.

It studies gradient-based RLVR over training trajectories, not one-shot zeroth-order perturbation selection. Its sparsity threshold is tied to bf16/numerical precision, so the exact sparsity values should not be transferred blindly. The result should steer our priors and evaluation design, not replace the dense-vs-LoRA baseline.

## Re-read Notes

The strongest project-relevant details are:

```text
1. AdamW's second-moment distribution is much less heterogeneous in RLVR than SFT.
2. AdamW momentum can be stale in online RL; the paper reports near-zero RL gradient/momentum alignment at one profiled step.
3. SGD needs much larger nominal learning rates because AdamW's effective per-parameter rates are far above its nominal LR.
4. SGD-style RLVR updates in Table 4 have about 99.84%-99.99% sparsity and 99%-energy effective ranks around 23.6-26.1 in Qwen settings.
```

For this lab, the paper should push us toward sparse-plus-low-rank search
families and skepticism toward accumulated stale directions. It should not make
us skip parity gates, because the paper studies trained RLVR gradients rather
than sampled zeroth-order perturbations.

## Implemented Probe Family

`sparse_low_rank_lora` now samples sparse LoRA factors while keeping the
expected `B @ A` entry variance matched to `factor_gaussian_lora`:

```text
A = sigma * mask_A * N(0, 1) / sqrt(density)
B = mask_B * N(0, 1) / sqrt(rank * density)
```

Available density names:

```text
sparse_low_rank_lora        # density 0.25
sparse_low_rank_lora_d0p125
sparse_low_rank_lora_d0p25
sparse_low_rank_lora_d0p5
```

This is a geometry probe, not a proven systems win. Standard vLLM LoRA serving
will still store dense adapter tensors unless we add sparse-aware packing or a
custom kernel, but the family directly tests whether the sparse factor prior
increases lucky-candidate density without changing the serving abstraction.
