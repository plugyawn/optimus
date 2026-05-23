#!/usr/bin/env bash
set -euo pipefail

DATA=${DATA:-data/countdown_generated_1200_seed20260507.json}
MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct}
OUT_ROOT=${OUT_ROOT:-results/optimus_gpu_suite}
POPULATIONS=${POPULATIONS:-"1024 4096"}
PROMPTS=${PROMPTS:-64}
HOLDOUT_PROMPTS=${HOLDOUT_PROMPTS:-256}
PROMOTE=${PROMOTE:-64}
RANK=${RANK:-8}
SIGMA=${SIGMA:-0.0075}
SEED=${SEED:-2468}
TARGETS=${TARGETS:-q_proj,v_proj}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-32}
CHUNK_ADAPTERS=${CHUNK_ADAPTERS:-8}
MAX_LORAS=${MAX_LORAS:-8}
MAX_CPU_LORAS=${MAX_CPU_LORAS:-8192}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-8}
SYSTEMS_OUT=${SYSTEMS_OUT:-results/report/optimus_systems}
BENCH_ADAPTERS=${BENCH_ADAPTERS:-8,16,32}
RUN_HALVING=${RUN_HALVING:-1}

export PYTHONUNBUFFERED=1
export VLLM_USAGE_STATS_ENABLED=${VLLM_USAGE_STATS_ENABLED:-0}
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-ERROR}
export XDG_CONFIG_HOME=${XDG_CONFIG_HOME:-/tmp/optimus-xdg-config}
mkdir -p "$OUT_ROOT" "$SYSTEMS_OUT" "$XDG_CONFIG_HOME"

halving_arg=()
if [[ "$RUN_HALVING" != "1" ]]; then
  halving_arg=(--skip-halving)
fi

optimus run-plan \
  --root "$OUT_ROOT" \
  --systems-out "$SYSTEMS_OUT" \
  --data "$DATA" \
  --model "$MODEL" \
  --populations "$(echo "$POPULATIONS" | tr ' ' ',')" \
  --prompts "$PROMPTS" \
  --holdout-prompts "$HOLDOUT_PROMPTS" \
  --promote "$PROMOTE" \
  --rank "$RANK" \
  --sigma "$SIGMA" \
  --seed "$SEED" \
  --targets "$TARGETS" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --chunk-adapters "$CHUNK_ADAPTERS" \
  --max-loras "$MAX_LORAS" \
  --max-cpu-loras "$MAX_CPU_LORAS" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --bench-adapters "$BENCH_ADAPTERS" \
  "${halving_arg[@]}" \
  --out "$OUT_ROOT/plan.json"

optimus run-suite \
  --root "$OUT_ROOT" \
  --systems-out "$SYSTEMS_OUT" \
  --data "$DATA" \
  --model "$MODEL" \
  --populations "$(echo "$POPULATIONS" | tr ' ' ',')" \
  --prompts "$PROMPTS" \
  --holdout-prompts "$HOLDOUT_PROMPTS" \
  --promote "$PROMOTE" \
  --rank "$RANK" \
  --sigma "$SIGMA" \
  --seed "$SEED" \
  --targets "$TARGETS" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --chunk-adapters "$CHUNK_ADAPTERS" \
  --max-loras "$MAX_LORAS" \
  --max-cpu-loras "$MAX_CPU_LORAS" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --bench-adapters "$BENCH_ADAPTERS" \
  "${halving_arg[@]}" \
  --execution-log "$OUT_ROOT/execution.json"

validate_halving_arg=()
if [[ "$RUN_HALVING" != "1" ]]; then
  validate_halving_arg=(--skip-halving)
fi

optimus validate-run \
  --root "$OUT_ROOT" \
  --systems-out "$SYSTEMS_OUT" \
  --populations "$(echo "$POPULATIONS" | tr ' ' ',')" \
  --bench-adapters "$BENCH_ADAPTERS" \
  "${validate_halving_arg[@]}" \
  --out "$OUT_ROOT/validation.json"
