#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen3-4B}
MODEL_TEMPLATE=${MODEL_TEMPLATE:-}
TASKS=${TASKS:-ifeval}
OUT_ROOT=${OUT_ROOT:-results/lighteval/population_sweep}
POPULATIONS=${POPULATIONS:-"128 256 512 1024 4096"}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
DATA_PARALLEL_SIZE=${DATA_PARALLEL_SIZE:-1}
PIPELINE_PARALLEL_SIZE=${PIPELINE_PARALLEL_SIZE:-}
DTYPE=${DTYPE:-bfloat16}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
MAX_MODEL_LENGTH=${MAX_MODEL_LENGTH:-4096}
MAX_SAMPLES=${MAX_SAMPLES:-}
CUSTOM_TASKS=${CUSTOM_TASKS:-}
RUN=${RUN:-0}

export PYTHONUNBUFFERED=1
export VLLM_USAGE_STATS_ENABLED=${VLLM_USAGE_STATS_ENABLED:-0}
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-ERROR}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export XDG_CONFIG_HOME=${XDG_CONFIG_HOME:-/tmp/optimus-xdg-config}
if [[ -n "${OPTIMUS_VLLM_ATTENTION_BACKEND:-}" ]]; then
  export VLLM_ATTENTION_BACKEND="$OPTIMUS_VLLM_ATTENTION_BACKEND"
fi
mkdir -p "$OUT_ROOT" "$XDG_CONFIG_HOME"

args=(
  --backend vllm
  --tasks "$TASKS"
  --model "$MODEL"
  --populations "$(echo "$POPULATIONS" | tr ' ' ',')"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --data-parallel-size "$DATA_PARALLEL_SIZE"
  --dtype "$DTYPE"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-model-length "$MAX_MODEL_LENGTH"
  --out-root "$OUT_ROOT"
  --plan-out "$OUT_ROOT/plan.json"
)

if [[ -n "$MODEL_TEMPLATE" ]]; then
  args+=(--model-template "$MODEL_TEMPLATE")
fi
if [[ -n "$PIPELINE_PARALLEL_SIZE" ]]; then
  args+=(--pipeline-parallel-size "$PIPELINE_PARALLEL_SIZE")
fi
if [[ -n "$MAX_SAMPLES" ]]; then
  args+=(--max-samples "$MAX_SAMPLES")
fi
if [[ -n "$CUSTOM_TASKS" ]]; then
  args+=(--custom-tasks "$CUSTOM_TASKS")
fi
if [[ "$RUN" == "1" ]]; then
  args+=(--run)
fi

optimus lighteval-sweep "${args[@]}"
