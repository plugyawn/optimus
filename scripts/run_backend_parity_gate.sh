#!/usr/bin/env bash
set -euo pipefail

DATA=${DATA:-data/countdown_generated_1200_seed20260507.json}
MODEL=${MODEL:-Qwen/Qwen3-4B}
OUT_ROOT=${OUT_ROOT:-results/backend_parity_gate}
FAMILY=${FAMILY:-factor_gaussian_lora}
POPULATION=${POPULATION:-64}
PROMPTS=${PROMPTS:-64}
HOLDOUT_PROMPTS=${HOLDOUT_PROMPTS:-8}
PROMOTE=${PROMOTE:-0}
RANK=${RANK:-8}
SIGMA=${SIGMA:-0.0075}
SEED=${SEED:-4242}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-32}
TARGETS=${TARGETS:-q_proj,v_proj}
HF_BATCH_SIZE=${HF_BATCH_SIZE:-16}
VLLM_MAX_LORAS=${VLLM_MAX_LORAS:-16}
VLLM_CHUNK_ADAPTERS=${VLLM_CHUNK_ADAPTERS:-16}
VLLM_PROMPT_INPUT=${VLLM_PROMPT_INPUT:-text}
ADAPTER_SAMPLE=${ADAPTER_SAMPLE:-16}

export PYTHONUNBUFFERED=1
export VLLM_USAGE_STATS_ENABLED=${VLLM_USAGE_STATS_ENABLED:-0}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export XDG_CONFIG_HOME=${XDG_CONFIG_HOME:-/tmp/optimus-xdg-config}
if [[ -n "${OPTIMUS_VLLM_ATTENTION_BACKEND:-}" ]]; then
  export VLLM_ATTENTION_BACKEND="$OPTIMUS_VLLM_ATTENTION_BACKEND"
fi
mkdir -p "$OUT_ROOT" "$XDG_CONFIG_HOME"

if [[ ! -f "$DATA" ]]; then
  optimus make-countdown-data --out "$DATA" --count 1200 --seed 20260507
fi

optimus peft-search \
  --out "$OUT_ROOT/peft" \
  --model "$MODEL" \
  --data "$DATA" \
  --family "$FAMILY" \
  --population "$POPULATION" \
  --prompts "$PROMPTS" \
  --holdout-prompts "$HOLDOUT_PROMPTS" \
  --promote "$PROMOTE" \
  --rank "$RANK" \
  --sigma "$SIGMA" \
  --seed "$SEED" \
  --targets "$TARGETS" \
  --batch-size "$HF_BATCH_SIZE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --stop-at-answer \
  --antithetic

optimus vllm-search \
  --out "$OUT_ROOT/vllm" \
  --model "$MODEL" \
  --data "$DATA" \
  --family "$FAMILY" \
  --population "$POPULATION" \
  --prompts "$PROMPTS" \
  --holdout-prompts "$HOLDOUT_PROMPTS" \
  --promote "$PROMOTE" \
  --rank "$RANK" \
  --sigma "$SIGMA" \
  --seed "$SEED" \
  --targets "$TARGETS" \
  --max-loras "$VLLM_MAX_LORAS" \
  --chunk-adapters "$VLLM_CHUNK_ADAPTERS" \
  --max-cpu-loras 4096 \
  --prompt-input "$VLLM_PROMPT_INPUT" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --stop-at-answer \
  --antithetic \
  --keep-adapters

optimus backend-parity-gate \
  --trusted "$OUT_ROOT/peft" \
  --candidate "$OUT_ROOT/vllm" \
  --trusted-name peft \
  --candidate-name vllm \
  --out "$OUT_ROOT/gate" \
  --adapter-sample "$ADAPTER_SAMPLE"
