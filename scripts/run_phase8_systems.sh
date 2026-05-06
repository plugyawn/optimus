#!/usr/bin/env bash
set -euo pipefail

DATA=${DATA:-data/countdown_generated_1200_seed20260507.json}
MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct}
PROMPTS=${PROMPTS:-64}
POPULATION=${POPULATION:-512}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-32}
RANK=${RANK:-8}
SIGMA=${SIGMA:-0.0075}
SEED=${SEED:-2468}
OUT_ROOT=${OUT_ROOT:-results/phase8_systems}
BENCH_EXTRA_ARGS=${BENCH_EXTRA_ARGS:-}
SEARCH_EXTRA_ARGS=${SEARCH_EXTRA_ARGS:-}

export PYTHONUNBUFFERED=1
export XDG_CONFIG_HOME=${XDG_CONFIG_HOME:-/tmp/randopt-xdg-config}
mkdir -p "$XDG_CONFIG_HOME"
export VLLM_USE_DEEP_GEMM=${VLLM_USE_DEEP_GEMM:-0}
export VLLM_DEEP_GEMM_WARMUP=${VLLM_DEEP_GEMM_WARMUP:-skip}

if [[ ! -f "$DATA" ]]; then
  python -m randopt_lora_lab.make_countdown_data --out "$DATA" --count 1200 --seed 20260507
fi

mkdir -p "$OUT_ROOT"

for adapters in 8 16 32; do
  python -m randopt_lora_lab.vllm_lora_bench \
    --out "$OUT_ROOT/bench_a${adapters}_p${PROMPTS}" \
    --model "$MODEL" \
    --data "$DATA" \
    --adapters "$adapters" \
    --prompts "$PROMPTS" \
    --rank "$RANK" \
    --sigma "$SIGMA" \
    --targets q_proj,v_proj \
    --max-loras "$adapters" \
    --max-cpu-loras 1024 \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --stop-at-answer \
    --preload \
    --mixed-batch \
    --skip-sequential \
    --no-include-base \
    $BENCH_EXTRA_ARGS
done

for chunk in 8 16 32; do
  python -m randopt_lora_lab.vllm_lora_search \
    --out "$OUT_ROOT/search_chunk${chunk}_p${POPULATION}" \
    --model "$MODEL" \
    --data "$DATA" \
    --prompts "$PROMPTS" \
    --holdout-prompts 8 \
    --population "$POPULATION" \
    --promote 0 \
    --rank "$RANK" \
    --sigma "$SIGMA" \
    --seed "$SEED" \
    --targets q_proj,v_proj \
    --max-loras "$chunk" \
    --chunk-adapters "$chunk" \
    --max-cpu-loras 2048 \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --stop-at-answer \
    --antithetic \
    $SEARCH_EXTRA_ARGS
done

python -m randopt_lora_lab.compare_backends \
  --trusted "$OUT_ROOT/search_chunk16_p${POPULATION}" \
  --candidate "$OUT_ROOT/search_chunk32_p${POPULATION}" \
  --trusted-name chunk16 \
  --candidate-name chunk32 \
  --out "$OUT_ROOT/chunk16_vs_chunk32" || true

python -m randopt_lora_lab.compare_backends \
  --trusted "$OUT_ROOT/search_chunk16_p${POPULATION}" \
  --candidate "$OUT_ROOT/search_chunk8_p${POPULATION}" \
  --trusted-name chunk16 \
  --candidate-name chunk8 \
  --out "$OUT_ROOT/chunk16_vs_chunk8" || true

python -m randopt_lora_lab.report --root results --out results/report
