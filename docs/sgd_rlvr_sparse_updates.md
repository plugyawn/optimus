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

## Cautions

The paper does not prove LoRA RandOpt parity.

It studies gradient-based RLVR over training trajectories, not one-shot zeroth-order perturbation selection. Its sparsity threshold is tied to bf16/numerical precision, so the exact sparsity values should not be transferred blindly. The result should steer our priors and evaluation design, not replace the dense-vs-LoRA baseline.
