# RandOpt Reproduction Boundary

This lab has two different baselines. They should not be mixed in reports.

## Official-Style RandOpt

The upstream Countdown script uses:

- model: `allenai/Olmo-3-7B-Instruct`
- train samples: `200`
- population: `5000`
- sigma values: `0.0005,0.001,0.002`
- top-K ratios: `0.04,0.01,0.05,0.1`
- max tokens: `1024`
- prompt: Countdown handler's reasoning prompt with the paper system message
- formatting: apply the tokenizer chat template for instruct/chat models
- perturbation support: all non-visual model parameters, not only selected
  attention projection matrices
- evaluation: select top candidates by train reward, then majority-vote over top-K candidates on test prompts
- Countdown voting: valid formulas vote by evaluated numeric result, not by raw formula string

Any claim about reproducing the paper must match these semantics, or explicitly
state every deviation.

Run the guardrail before making that claim:

```bash
python -m randopt_lora_lab.reproduction_audit \
  --run results/YOUR_RUN_DIR \
  --out results/YOUR_RUN_DIR/reproduction_audit
```

The audit must pass for an official-style reproduction claim. A failed audit can
still describe a useful local parity panel, but it must not be presented as a
paper reproduction.

## Corrected Local Parity Panel

The local dense-vs-LoRA panel is a systems and geometry parity test, not an
official paper reproduction. It may use a smaller model, smaller population, and
answer-only prompt, but it must preserve the core selection semantics:

- sample a shared candidate panel for dense and LoRA arms
- support multiple sigma values in the same population
- rank candidates only on the screen split
- evaluate enough promoted candidates to cover every requested ensemble K
- report `ensemble_holdout`, not only `top_holdout`
- vote by valid numeric Countdown result
- keep `screen_holdout_overlap == 0`
- use no repeated semantic Countdown examples for quality claims

`top_holdout` is only a single-candidate diagnostic. A quality claim based on
RandOpt-style search must use `ensemble_holdout`.

For an official-style prompt on the local dense/LoRA harness, use:

```bash
python -m randopt_lora_lab.experiments search \
  --out results/paper_style_dense_local \
  --model Qwen/Qwen2.5-3B-Instruct \
  --perturbation-backend dense \
  --family dense_gaussian \
  --targets all_params \
  --dense-noise-mode paper \
  --population 128 \
  --prompts 64 \
  --holdout-prompts 256 \
  --sigma-values 0.0005,0.001,0.002 \
  --promote 32 \
  --ensemble-ratios 0.04,0.01,0.05,0.1 \
  --prompt-variant paper \
  --use-chat-template \
  --max-new-tokens 1024
```

## Required Gates

Before claiming LoRA is as good as dense Gaussian RandOpt:

- `best_ensemble_holdout_exact` for LoRA must match or beat dense under the same
  population, sigma panel, screen split, holdout split, prompt, token cap, and
  generation backend.
- parity reports must pass rank, top-K overlap, selected-regret, speed, and
  ensemble-quality gates.
- malformed, cap-hit, and answer-closed rates must be reported for selected
  candidates and the ensemble prompt rows.
- vLLM may only be used as a selector after a backend parity gate shows that the
  same candidate panel has trusted rank correlation and top-K overlap against
  the PEFT/HF path.
