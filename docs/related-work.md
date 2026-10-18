# Related work

## The method

- **RandOpt / Neural Thickets.** Gan and Isola,
  [*Neural Thickets: Diverse Task Experts Are Dense Around Pretrained Weights*](https://arxiv.org/abs/2603.12228)
  (2026). Random perturbations of a pretrained model, selected by task reward
  and ensembled, improve task accuracy without gradients. Optimus implements
  this search and changes the perturbation family so it can be batched.
- **Zeroth-order fine-tuning.** Malladi et al.,
  [*Fine-Tuning Language Models with Just Forward Passes*](https://arxiv.org/abs/2305.17333)
  (MeZO, 2023), and evolution-strategies fine-tuning of LLMs at scale (Qiu et
  al., 2025). Both estimate updates from forward passes only. Optimus does not
  estimate a gradient; it selects among candidates, so its cost is dominated by
  evaluating many candidates rather than by many sequential steps.
- **Low-rank evolution strategies.** EGGROLL (*Evolution Strategies at the
  Hyperscale*, 2025) uses low-rank perturbations so a large population can be
  evaluated in batched inference. The motivation is the same; Optimus
  additionally ties the low-rank input side to activation bases measured at
  each site.

## Batched low-rank serving

- **Punica.** Chen et al.,
  [*Punica: Multi-Tenant LoRA Serving*](https://arxiv.org/abs/2310.18547)
  (2023). Introduced SGMV (segmented gather matrix-vector multiplication) and
  BGMV kernels, so requests for different LoRA adapters share one batch: a
  *shrink* into rank `r`, then an *expand*, with rows grouped by adapter.
- **S-LoRA.** Sheng et al.,
  [*S-LoRA: Serving Thousands of Concurrent LoRA Adapters*](https://arxiv.org/abs/2311.03285)
  (2023). Unified paging for adapter weights and KV cache, so thousands of
  adapters can be served from one base model.
- **vLLM.** Kwon et al.,
  [*Efficient Memory Management for Large Language Model Serving with PagedAttention*](https://arxiv.org/abs/2309.06180)
  (2023). vLLM's LoRA support carries Triton descendants of the Punica kernels
  (`lora_shrink`, `lora_expand`). Optimus's first hook path called these
  directly; its LoRA baseline uses vLLM's multi-LoRA batching as is.

How Optimus differs from adapter serving:

| | multi-LoRA serving | Optimus subspace search |
| --- | --- | --- |
| shrink factor | per adapter (`A_i`) | per activation site (`Q_s`), shared by every candidate |
| expand factor | per adapter (`B_i`), loaded from memory | per candidate (`G_c`), generated in the kernel from a seed |
| what limits the population | adapter memory and slots | compute |
| KV cache | can share prefixes across adapters only with care | per candidate, prefix caching off |

## Determinism

- **Batch invariance.** He and Thinking Machines Lab,
  [*Defeating Nondeterminism in LLM Inference*](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
  (2025). Kernel reduction order that depends on batch composition makes greedy
  decoding batch-dependent. This is the likely cause of the remaining
  candidate-batch-size replay mismatch in Optimus (1 score in 128) and is the
  reference for closing it.
