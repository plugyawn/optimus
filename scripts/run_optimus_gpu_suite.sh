#!/usr/bin/env bash
set -euo pipefail

DATA=${DATA:-data/countdown_generated_1200_seed20260507.json}
MODEL=${MODEL:-Qwen/Qwen3-4B}
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
PROMPT_VARIANTS=${PROMPT_VARIANTS:-default}
PROMPT_INPUT=${PROMPT_INPUT:-text}
USE_CHAT_TEMPLATE=${USE_CHAT_TEMPLATE:-0}
MAX_BASE_MALFORMED_FOR_SELECTION=${MAX_BASE_MALFORMED_FOR_SELECTION:-0.05}
MAX_BASE_CAP_HIT_FOR_SELECTION=${MAX_BASE_CAP_HIT_FOR_SELECTION:-0.05}
MIN_SELECTION_PROMPT_VARIANTS=${MIN_SELECTION_PROMPT_VARIANTS:-1}
REQUIRE_ALL_PROMPT_VARIANTS_VALID=${REQUIRE_ALL_PROMPT_VARIANTS_VALID:-0}
CHUNK_ADAPTERS=${CHUNK_ADAPTERS:-32}
MAX_LORAS=${MAX_LORAS:-32}
MAX_CPU_LORAS=${MAX_CPU_LORAS:-8192}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
SYSTEMS_OUT=${SYSTEMS_OUT:-results/report/optimus_systems}
BENCH_ADAPTERS=${BENCH_ADAPTERS:-8,16,32}
RUN_HALVING=${RUN_HALVING:-1}
KEEP_ADAPTERS=${KEEP_ADAPTERS:-0}

export PYTHONUNBUFFERED=1
export VLLM_USAGE_STATS_ENABLED=${VLLM_USAGE_STATS_ENABLED:-0}
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-ERROR}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export XDG_CONFIG_HOME=${XDG_CONFIG_HOME:-/tmp/optimus-xdg-config}
if [[ -n "${OPTIMUS_VLLM_ATTENTION_BACKEND:-}" ]]; then
  export VLLM_ATTENTION_BACKEND="$OPTIMUS_VLLM_ATTENTION_BACKEND"
fi
mkdir -p "$OUT_ROOT" "$SYSTEMS_OUT" "$XDG_CONFIG_HOME"

halving_arg=()
if [[ "$RUN_HALVING" != "1" ]]; then
  halving_arg=(--skip-halving)
fi
artifact_arg=()
if [[ "$KEEP_ADAPTERS" == "1" ]]; then
  artifact_arg=(--keep-adapters)
fi
prompt_contract_args=(
  --prompt-variants "$PROMPT_VARIANTS"
  --prompt-input "$PROMPT_INPUT"
  --max-base-malformed-for-selection "$MAX_BASE_MALFORMED_FOR_SELECTION"
  --max-base-cap-hit-for-selection "$MAX_BASE_CAP_HIT_FOR_SELECTION"
  --min-selection-prompt-variants "$MIN_SELECTION_PROMPT_VARIANTS"
)
case "$USE_CHAT_TEMPLATE" in
  1|true|TRUE|yes|YES) prompt_contract_args+=(--use-chat-template) ;;
esac
case "$REQUIRE_ALL_PROMPT_VARIANTS_VALID" in
  1|true|TRUE|yes|YES) prompt_contract_args+=(--require-all-prompt-variants-valid) ;;
esac

vllm_runtime_args=()
case "${ENABLE_PREFIX_CACHING:-}" in
  1|true|TRUE|yes|YES) vllm_runtime_args+=(--enable-prefix-caching) ;;
  0|false|FALSE|no|NO) vllm_runtime_args+=(--no-enable-prefix-caching) ;;
esac
case "${ENABLE_CHUNKED_PREFILL:-}" in
  1|true|TRUE|yes|YES) vllm_runtime_args+=(--enable-chunked-prefill) ;;
  0|false|FALSE|no|NO) vllm_runtime_args+=(--no-enable-chunked-prefill) ;;
esac
if [[ -n "${KV_CACHE_DTYPE:-}" ]]; then
  vllm_runtime_args+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
fi
for item in ${VLLM_KWARGS:-}; do
  vllm_runtime_args+=(--vllm-kwarg "$item")
done

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
  "${prompt_contract_args[@]}" \
  --chunk-adapters "$CHUNK_ADAPTERS" \
  --max-loras "$MAX_LORAS" \
  --max-cpu-loras "$MAX_CPU_LORAS" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --bench-adapters "$BENCH_ADAPTERS" \
  "${vllm_runtime_args[@]}" \
  "${artifact_arg[@]}" \
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
  "${prompt_contract_args[@]}" \
  --chunk-adapters "$CHUNK_ADAPTERS" \
  --max-loras "$MAX_LORAS" \
  --max-cpu-loras "$MAX_CPU_LORAS" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --bench-adapters "$BENCH_ADAPTERS" \
  "${vllm_runtime_args[@]}" \
  "${artifact_arg[@]}" \
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
