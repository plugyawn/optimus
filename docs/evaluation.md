# Optimus Evaluation

Optimus separates candidate proposal from trusted evaluation.

The high-throughput path uses vLLM adapter swapping to screen thousands of LoRA
candidates. That path is optimized for proposal throughput and selector
diagnostics.

The trusted-eval path uses HF/PEFT, dense-reference checks, backend parity gates,
and LightEval. LightEval is the standard harness for benchmark/task execution,
HF Hub dataset loading, saved sample-level details, and fast backends such as
vLLM.

## LightEval

Install the eval extra:

```bash
python -m pip install -e ".[eval]"
```

Install the serving extra as well when the LightEval backend is `vllm`:

```bash
python -m pip install -e ".[eval,serving]"
```

Plan a LightEval run without importing or executing LightEval:

```bash
optimus lighteval \
  --backend vllm \
  --model Qwen/Qwen2.5-3B-Instruct \
  --tensor-parallel-size 4 \
  --tasks ifeval \
  --out results/lighteval/ifeval_qwen25_3b \
  --plan-out results/lighteval/ifeval_qwen25_3b/plan.json
```

Execute the same run by adding `--run`.

Use LightEval for final confirmation of selected base/LoRA candidates, not for
the full P1024/P4096 candidate-screening loop. The search loop requires
per-candidate adapter swapping and candidate manifests; LightEval is the
right tool once Optimus has selected a small set of candidates to evaluate on a
standard task or a custom HF-hosted task.

## Required Evidence

Publication-grade selector evidence should include:

- screen-only candidate selection;
- HF/PEFT or LightEval confirmation for the selected candidates;
- sample-level details saved for every evaluated example;
- confidence intervals or paired significance tests;
- no headline selector claim based on holdout-oracle selection.
