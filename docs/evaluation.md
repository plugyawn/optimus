# Optimus Evaluation

Optimus separates candidate proposal from trusted evaluation.

The high-throughput vLLM path screens LoRA perturbations through adapter serving
and is the planned production substrate for subspace search. Dense perturbations
use the trusted Transformers path because they mutate model weights directly.

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
artifact. Lazy top-K subspace ensembles are evaluated with Optimus-native
sample-level evaluation in v1; LightEval is valid for a single-candidate export,
merged checkpoint, or distilled ensemble artifact unless a later PR adds a
direct lazy-ensemble LightEval runtime. LightEval is not the full P1024/P4096
candidate-screening loop; the sweep command expects model/path templates for
already materialized selected states.

## Required Evidence

Publication-grade selector evidence should include:

- screen-only candidate selection;
- Transformers, dense-reference, Optimus-native lazy-ensemble, or LightEval
  confirmation for the selected state, matched to the artifact type;
- sample-level details saved for every evaluated example;
- confidence intervals or paired significance tests;
- no headline selector claim based on holdout-oracle selection.
