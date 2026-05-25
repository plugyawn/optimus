#!/usr/bin/env bash
set -euo pipefail

DATA=${DATA:-data/countdown_generated_1200_seed20260507.json}
MODEL=${MODEL:-Qwen/Qwen3-4B}
BACKEND=${BACKEND:-vllm}
METHOD=${METHOD:-lora}
OUT_ROOT=${OUT_ROOT:-results/optimus_gpu_suite}
POPULATIONS=${POPULATIONS:-"1024 4096"}
PROMPTS=${PROMPTS:-64}
HOLDOUT_PROMPTS=${HOLDOUT_PROMPTS:-256}
PROMOTE=${PROMOTE:-64}
RANK=${RANK:-8}
SIGMA=${SIGMA:-0.0075}
BASIS_RANK=${BASIS_RANK:-128}
BASIS_PROMPTS=${BASIS_PROMPTS:-32}
TARGET_PRESET=${TARGET_PRESET:-transformer-linears}
LAYERS=${LAYERS:-all}
BASIS_CENTERING=${BASIS_CENTERING:-none}
BASIS_TOKEN_SOURCE=${BASIS_TOKEN_SOURCE:-prefill}
SCALE_MODE=${SCALE_MODE:-relative-output-rms}
RHO_GRID=${RHO_GRID:-0.002,0.005,0.01,0.02}
SIGMA_W_GRID=${SIGMA_W_GRID:-}
BUDGET_POLICY=${BUDGET_POLICY:-per-block-equal}
BASIS_KIND=${BASIS_KIND:-activation-svd}
TOP_K_GRID=${TOP_K_GRID:-1,4,8,16}
CANDIDATE_BATCH_SIZE=${CANDIDATE_BATCH_SIZE:-auto}
KERNEL=${KERNEL:-torch}
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
RUN_HALVING=${RUN_HALVING:-0}
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
if [[ "$RUN_HALVING" == "1" ]]; then
  halving_arg=(--run-halving)
else
  halving_arg=(--skip-halving)
fi
artifact_arg=()
if [[ "$KEEP_ADAPTERS" == "1" && "$METHOD" == "lora" ]]; then
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

method_args=()
if [[ "$METHOD" == "subspace" ]]; then
  method_args=(
    --basis-rank "$BASIS_RANK"
    --basis-prompts "$BASIS_PROMPTS"
    --target-preset "$TARGET_PRESET"
    --layers "$LAYERS"
    --basis-centering "$BASIS_CENTERING"
    --basis-token-source "$BASIS_TOKEN_SOURCE"
    --scale-mode "$SCALE_MODE"
    --budget-policy "$BUDGET_POLICY"
    --basis-kind "$BASIS_KIND"
    --top-k-grid "$TOP_K_GRID"
    --candidate-batch-size "$CANDIDATE_BATCH_SIZE"
    --kernel "$KERNEL"
  )
  if [[ "${MATCH_SCREEN_TO_HOLDOUT_BASE_EXACT:-0}" == "1" ]]; then
    method_args+=(--match-screen-to-holdout-base-exact)
  fi
  if [[ -n "${SCREEN_POOL_PROMPTS:-}" ]]; then
    method_args+=(--screen-pool-prompts "$SCREEN_POOL_PROMPTS")
  fi
  if [[ "$SCALE_MODE" == "projected-dense" ]]; then
    method_args+=(--sigma-w-grid "$SIGMA_W_GRID")
  else
    method_args+=(--rho-grid "$RHO_GRID")
  fi
else
  method_args=(
    --rank "$RANK"
    --sigma "$SIGMA"
    --targets "$TARGETS"
    --chunk-adapters "$CHUNK_ADAPTERS"
    --max-loras "$MAX_LORAS"
    --max-cpu-loras "$MAX_CPU_LORAS"
  )
fi

optimus run-plan \
  --root "$OUT_ROOT" \
  --systems-out "$SYSTEMS_OUT" \
  --data "$DATA" \
  --model "$MODEL" \
  --backend "$BACKEND" \
  --method "$METHOD" \
  --populations "$(echo "$POPULATIONS" | tr ' ' ',')" \
  --prompts "$PROMPTS" \
  --holdout-prompts "$HOLDOUT_PROMPTS" \
  --promote "$PROMOTE" \
  --seed "$SEED" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  "${prompt_contract_args[@]}" \
  "${method_args[@]}" \
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
  --backend "$BACKEND" \
  --method "$METHOD" \
  --populations "$(echo "$POPULATIONS" | tr ' ' ',')" \
  --prompts "$PROMPTS" \
  --holdout-prompts "$HOLDOUT_PROMPTS" \
  --promote "$PROMOTE" \
  --seed "$SEED" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  "${prompt_contract_args[@]}" \
  "${method_args[@]}" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --bench-adapters "$BENCH_ADAPTERS" \
  "${vllm_runtime_args[@]}" \
  "${artifact_arg[@]}" \
  "${halving_arg[@]}" \
  --execution-log "$OUT_ROOT/execution.json"

validate_halving_arg=()
if [[ "$RUN_HALVING" == "1" ]]; then
  validate_halving_arg=(--run-halving)
else
  validate_halving_arg=(--skip-halving)
fi

optimus validate-run \
  --root "$OUT_ROOT" \
  --systems-out "$SYSTEMS_OUT" \
  --backend "$BACKEND" \
  --method "$METHOD" \
  --populations "$(echo "$POPULATIONS" | tr ' ' ',')" \
  --bench-adapters "$BENCH_ADAPTERS" \
  "${validate_halving_arg[@]}" \
  --out "$OUT_ROOT/validation.json" \
  --strict
