# Backend Parity Gate P16

Date: 2026-05-07

Pod: `7d357aa41b494ae3b1c9bed6a7f1c7cc`, 1x A100 80GB, terminated after artifact pull.

Run:

```bash
OUT_ROOT=results/backend_parity_gate_p16 \
FAMILY=factor_gaussian_lora \
POPULATION=16 \
PROMPTS=16 \
HOLDOUT_PROMPTS=8 \
PROMOTE=0 \
RANK=8 \
SIGMA=0.0075 \
SEED=4242 \
MAX_NEW_TOKENS=32 \
VLLM_MAX_LORAS=8 \
VLLM_CHUNK_ADAPTERS=8 \
ADAPTER_SAMPLE=4 \
scripts/run_backend_parity_gate.sh
```

## Verdict

The gate failed. This means vLLM is not yet allowed to be the quality selector
for this perturbation family.

| Gate | Result |
| --- | ---: |
| Protocol metadata | pass |
| Base rows present | pass |
| Adapter tensor parity | pass |
| Ranking parity | fail |

The important split is:

```text
adapter tensor checks: 576/576 passed on 4 sampled adapters
Spearman(PEFT score, vLLM score): -0.181164
top-8 overlap: 7/8
selected regret vs PEFT: 0.125
```

PEFT selected:

```text
factor_gaussian_lora:seed509771609:s0.0075:sign-1
PEFT exact: 0.125
vLLM exact: 0.0
```

vLLM selected:

```text
factor_gaussian_lora:seed1019282515:s0.0075:sign1
vLLM exact: 0.0625
PEFT exact: 0.0
```

## Systems Signal

The accelerated path is still materially faster in this smoke:

```text
PEFT candidate/sec: 0.717830
vLLM candidate/sec: 6.616220
speedup: 9.22x
```

Cold vLLM load was `37.895s`; adapter build was `1.385s`.

## Interpretation

This is no longer a candidate-key or adapter-file materialization bug: sampled
vLLM adapter tensors match the canonical materializer. The remaining mismatch is
downstream of tensor materialization, likely backend generation semantics,
adapter application/scaling semantics, or score sparsity/tie instability on a
tiny screen.

Next diagnosis should compare PEFT and vLLM logits or deterministic next-token
distributions for base, zero adapter, and the two disagreeing candidates before
running larger vLLM searches.
