# Optimus

Optimus is a research library for zeroth-order post-training of language
models. It samples many small perturbations of a pretrained model, scores each
one on a task, and keeps the best. Evaluating candidates is the expensive part,
so Optimus runs thousands of them through a single vLLM model copy, many
candidates per batch.

![Systems view of an Optimus search](assets/system.svg)

## Start here

- [How it works](how-it-works.md): the perturbation inside one decoder layer,
  then inside the serving stack.
- [Results](results.md): where time goes, the in-kernel random field, and
  end-to-end throughput.
- [Quickstart](quickstart.md): install and run a search.
- [Related work](related-work.md): RandOpt, Punica and S-LoRA, zeroth-order
  fine-tuning.

## Status

| path | state |
| --- | --- |
| Dense and LoRA search through Transformers | supported; the trusted reference |
| LoRA search through vLLM, many adapters per batch | supported; the baseline |
| Subspace search through vLLM | runs from a forward hook with Triton kernels, measured to population 1,024 |
| Fused subspace custom op | not written yet |
| Strict hook-versus-adapter logprob parity | open |

## Reference

| document | purpose |
| --- | --- |
| [API](api.md) | Supported Python packages, CLI commands, and import discipline. |
| [Design](optimus_design.md) | Package structure, API boundaries, and completion criteria. |
| [Subspace kernel design](full_model_lazy_kernel_design.md) | Source of truth for transformer-linear subspace search over vLLM. |
| [Subspace roadmap](subspace_implementation_roadmap.md) | Phases and acceptance gates for subspace search. |
| [GPU suite](gpu_suite.md) | Required P1024/P4096 workloads and output contracts. |
| [Evaluation](evaluation.md) | LightEval confirmation and trusted-eval policy. |
| [Prime GPU runbook](prime_gpu_runbook.md) | Launch, sync, validation, and cleanup on Prime pods. |
| [Release checklist](release_checklist.md) | Identity, evidence, systems, and upstreaming gates. |
