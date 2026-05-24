#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen3-4B}
DATA=${DATA:-data/countdown_generated_1200_seed20260507.json}
POPULATIONS=${POPULATIONS:-"128 256 512 1024 4096"}
TASKS=${TASKS:-ifeval}
RESULTS_ROOT=${RESULTS_ROOT:-results}
SEARCH_ROOT=${SEARCH_ROOT:-$RESULTS_ROOT/optimus_gpu_suite}
SYSTEMS_OUT=${SYSTEMS_OUT:-$RESULTS_ROOT/report/optimus_systems}
MATERIALIZED_ROOT=${MATERIALIZED_ROOT:-$RESULTS_ROOT/materialized}
LIGHTEVAL_ROOT=${LIGHTEVAL_ROOT:-$RESULTS_ROOT/lighteval}
REPORT_OUT=${REPORT_OUT:-$LIGHTEVAL_ROOT/report}
MATERIALIZE_MODE=${MATERIALIZE_MODE:-merged}
MATERIALIZE_SELECTION=${MATERIALIZE_SELECTION:-top_screen}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-2}
DATA_PARALLEL_SIZE=${DATA_PARALLEL_SIZE:-1}
DTYPE=${DTYPE:-bfloat16}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
MAX_MODEL_LENGTH=${MAX_MODEL_LENGTH:-4096}
RUN_HALVING=${RUN_HALVING:-0}
KEEP_ADAPTERS=${KEEP_ADAPTERS:-1}
CUSTOM_TASKS=${CUSTOM_TASKS:-}
MAX_SAMPLES=${MAX_SAMPLES:-}

export PYTHONUNBUFFERED=1
export VLLM_USAGE_STATS_ENABLED=${VLLM_USAGE_STATS_ENABLED:-0}
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-ERROR}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export XDG_CONFIG_HOME=${XDG_CONFIG_HOME:-/tmp/optimus-xdg-config}
if [[ -n "${OPTIMUS_VLLM_ATTENTION_BACKEND:-}" ]]; then
  export VLLM_ATTENTION_BACKEND="$OPTIMUS_VLLM_ATTENTION_BACKEND"
fi
mkdir -p "$RESULTS_ROOT" "$SEARCH_ROOT" "$MATERIALIZED_ROOT" "$LIGHTEVAL_ROOT" "$XDG_CONFIG_HOME"

KEEP_ADAPTERS="$KEEP_ADAPTERS" \
RUN_HALVING="$RUN_HALVING" \
MODEL="$MODEL" \
DATA="$DATA" \
OUT_ROOT="$SEARCH_ROOT" \
SYSTEMS_OUT="$SYSTEMS_OUT" \
POPULATIONS="$POPULATIONS" \
TENSOR_PARALLEL_SIZE="$TENSOR_PARALLEL_SIZE" \
scripts/run_optimus_gpu_suite.sh

optimus materialize-selected \
  --root "$SEARCH_ROOT" \
  --out-root "$MATERIALIZED_ROOT" \
  --selection "$MATERIALIZE_SELECTION" \
  --mode "$MATERIALIZE_MODE" \
  --model "$MODEL"

base_args=(
  --backend vllm
  --tasks "$TASKS"
  --model "$MODEL"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --data-parallel-size "$DATA_PARALLEL_SIZE"
  --dtype "$DTYPE"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-model-length "$MAX_MODEL_LENGTH"
  --out "$LIGHTEVAL_ROOT/base"
  --plan-out "$LIGHTEVAL_ROOT/base/plan.json"
  --run
)

if [[ -n "$CUSTOM_TASKS" ]]; then
  base_args+=(--custom-tasks "$CUSTOM_TASKS")
fi
if [[ -n "$MAX_SAMPLES" ]]; then
  base_args+=(--max-samples "$MAX_SAMPLES")
fi

optimus lighteval "${base_args[@]}"

sweep_args=(
  --backend vllm
  --tasks "$TASKS"
  --model "$MODEL"
  --model-template "$MATERIALIZED_ROOT/p{population}"
  --populations "$(echo "$POPULATIONS" | tr ' ' ',')"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --data-parallel-size "$DATA_PARALLEL_SIZE"
  --dtype "$DTYPE"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-model-length "$MAX_MODEL_LENGTH"
  --out-root "$LIGHTEVAL_ROOT/population_sweep"
  --plan-out "$LIGHTEVAL_ROOT/population_sweep/plan.json"
  --continue-on-error
  --run
)

if [[ -n "$CUSTOM_TASKS" ]]; then
  sweep_args+=(--custom-tasks "$CUSTOM_TASKS")
fi
if [[ -n "$MAX_SAMPLES" ]]; then
  sweep_args+=(--max-samples "$MAX_SAMPLES")
fi

optimus lighteval-sweep "${sweep_args[@]}"
optimus lighteval-report --root "$LIGHTEVAL_ROOT" --out "$REPORT_OUT"
