# Full Gaussian Parity Gate

## Current Claim Status

We have not shown that rank-8 LoRA perturbation search is as powerful as dense Gaussian RandOpt.

The structural reason is simple. A LoRA adapter changes a target matrix by

```text
Delta W = B @ A
```

where `A` is `[rank, in_features]` and `B` is `[out_features, rank]`. Therefore `Delta W` has rank at most `rank`. A dense iid Gaussian perturbation to the same matrix is full-rank almost surely, with rank `min(out_features, in_features)`.

For Qwen2.5-3B-style `q_proj,v_proj` rank-8 LoRA:

```text
q_proj dense shape: 2048 x 2048
v_proj dense shape: 256 x 2048
rank-8 q_proj LoRA parameter fraction: 0.78125%
rank-8 v_proj LoRA parameter fraction: 3.515625%
combined q/v LoRA parameter fraction: about 1.085%
combined q/v rank fraction: about 0.6944%
```

So exact expressivity parity with arbitrary dense Gaussian RandOpt is impossible at low rank. The right target is search-utility parity under a matched budget, or exact parity with the best rank-`r` projection of a dense Gaussian perturbation.

## Three Perturbation Families

Use precise names in reports:

```text
dense_gaussian
  Full iid Gaussian delta directly added to each selected weight matrix.

projected_gaussian_rank_r
  Sample dense_gaussian, take its best rank-r SVD projection, then factor it as LoRA B @ A.
  This is the fair bridge between dense Gaussian geometry and a serveable LoRA adapter.

factor_gaussian_lora
  Current method: sample Gaussian factors A and B directly.
  This is not the same distribution as projected_gaussian_rank_r.
```

## Parity Requirements

A LoRA family can be called parity-equivalent to dense Gaussian only if it passes all gates below.

```text
Stability
  Same candidate panels across seeds.
  Report Spearman/Kendall, top-k overlap, selected-candidate regret, and seed-to-seed variance.

Drift
  Match perturbation scale by measured drift, not nominal sigma alone.
  Report update norm, logit L2, true or approximate token-distribution KL, cap-hit rate, malformed rate, and output-token distribution.

Eval speed
  Report candidate/sec, prompt-evals/sec, adapter build time, cold-start time, warm steady-state time, and GPU memory.

Convenience
  Report whether candidates can be represented as portable adapter files, hot-swapped, batched, replayed from seed, and audited after deletion.

Robustness
  Use disjoint screen/validation/test prompts with no repeated examples.
  Include base, zero-adapter, random controls, same-mean scale controls, and parser/manual audit panels.
```

## Next GPU Baseline

Run a small but clean baseline before another large search:

```text
model: same Qwen checkpoint
targets: q_proj,v_proj
rank: 8 and 32
families: dense_gaussian, projected_gaussian_rank_r, factor_gaussian_lora
population: 64 or 128
screen prompts: 64
validation prompts: 256
test prompts: 512
decode: answer-only, stop-at-answer, fixed max_new_tokens
scale matching: grid sigma, then compare at matched logit drift / KL bucket
```

Acceptance:

```text
1. projected_gaussian_rank_r reconstructs the SVD projection exactly as B @ A.
2. factor_gaussian_lora matches or beats projected_gaussian_rank_r at the same drift and wall-clock budget.
3. factor_gaussian_lora matches or beats dense_gaussian selected holdout lift within paired uncertainty.
4. cap-hit, malformed, and token-count distributions do not regress.
5. vLLM/SGLang adapter serving preserves the same candidate ranking as the trusted reference path.
```

If rank-8 fails but rank-32 passes, the result is still useful: the question becomes the minimum serveable rank needed for dense-Gaussian search-utility parity.
