# Optimus Evaluation

Optimus separates candidate proposal from trusted evaluation.

The high-throughput vLLM path screens thousands of LoRA perturbations through
adapter swapping. Dense perturbations use the trusted Transformers path because
they mutate model weights directly.

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
  --model Qwen/Qwen3-4B \
  --data-parallel-size 8 \
  --max-model-length 4096 \
  --tasks ifeval \
  --out results/lighteval/ifeval_qwen3_4b \
  --plan-out results/lighteval/ifeval_qwen3_4b/plan.json
```

Execute the same run by adding `--run`.

Plan a population-labelled sweep for selected/materialized checkpoints:

```bash
optimus lighteval-sweep \
  --backend vllm \
  --tasks ifeval \
  --model-template results/materialized/p{population} \
  --populations 128,256,512,1024,4096 \
  --data-parallel-size 8 \
  --max-model-length 4096 \
  --out-root results/lighteval/population_sweep \
  --plan-out results/lighteval/population_sweep/plan.json
```

Use LightEval for final confirmation of an externally materialized model path,
for example a base model, merged adapter checkpoint, or other final model
artifact. It is not the full P1024/P4096 candidate-screening loop; the sweep
command expects the caller to provide model/path templates for already
materialized selected states.

## Required Evidence

Publication-grade selector evidence should include:

- screen-only candidate selection;
- Transformers, dense-reference, or LightEval confirmation for the selected
  materialized model state;
- sample-level details saved for every evaluated example;
- confidence intervals or paired significance tests;
- no headline selector claim based on holdout-oracle selection.
