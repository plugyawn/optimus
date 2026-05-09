#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run vLLM shortlist screening, PEFT-confirm only top-K candidates, and compare
the confirmed shortlist against a full dense Gaussian reference.

Configure with environment variables, for example:

  OUT_ROOT=results/vllm_shortlist_sparse_d0p125_p64 \
  FAMILY=sparse_low_rank_lora_d0p125 \
  POPULATION=64 \
  SHORTLIST_K=8 \
  scripts/run_vllm_shortlist_confirmation.sh

EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ "$#" -ne 0 ]]; then
  usage >&2
  echo "unexpected positional arguments: $*" >&2
  exit 2
fi

DATA=${DATA:-data/countdown_generated_1200_seed20260507.json}
MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct}
OUT_ROOT=${OUT_ROOT:-results/vllm_shortlist_sparse_d0p125_p64}
PYTHON=${PYTHON:-python}

FAMILY=${FAMILY:-sparse_low_rank_lora_d0p125}
POPULATION=${POPULATION:-64}
PROMPTS=${PROMPTS:-64}
HOLDOUT_PROMPTS=${HOLDOUT_PROMPTS:-256}
VLLM_HOLDOUT_PROMPTS=${VLLM_HOLDOUT_PROMPTS:-8}
SHORTLIST_K=${SHORTLIST_K:-8}
RANK=${RANK:-32}
SIGMA=${SIGMA:-0.001}
SIGMA_VALUES=${SIGMA_VALUES:-0.0005,0.001,0.002}
SEED=${SEED:-20260507}
TARGETS=${TARGETS:-q_proj,v_proj}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
HF_BATCH_SIZE=${HF_BATCH_SIZE:-32}
ENSEMBLE_KS=${ENSEMBLE_KS:-1,4,8}

RUN_DENSE=${RUN_DENSE:-1}
RUN_VLLM=${RUN_VLLM:-1}
RUN_CONFIRM=${RUN_CONFIRM:-1}
RUN_REPORT=${RUN_REPORT:-1}
RUN_PROVENANCE_AUDIT=${RUN_PROVENANCE_AUDIT:-1}

VLLM_PROMPT_INPUT=${VLLM_PROMPT_INPUT:-token_ids}
VLLM_PROMPT_VARIANTS=${VLLM_PROMPT_VARIANTS:-${PROMPT_VARIANTS:-default,reordered,xml}}
VLLM_SCORE_MODE=${VLLM_SCORE_MODE:-robust_mean}
VLLM_MIN_SELECTION_PROMPT_VARIANTS=${VLLM_MIN_SELECTION_PROMPT_VARIANTS:-2}
VLLM_MAX_BASE_MALFORMED=${VLLM_MAX_BASE_MALFORMED:-0.05}
VLLM_MAX_BASE_CAP_HIT=${VLLM_MAX_BASE_CAP_HIT:-0.05}
VLLM_REQUIRE_ALL_PROMPT_VARIANTS_VALID=${VLLM_REQUIRE_ALL_PROMPT_VARIANTS_VALID:-1}
VLLM_MALFORMED_PENALTY=${VLLM_MALFORMED_PENALTY:-1.0}
VLLM_CAP_HIT_PENALTY=${VLLM_CAP_HIT_PENALTY:-1.0}
VLLM_MAX_LORAS=${VLLM_MAX_LORAS:-16}
VLLM_CHUNK_ADAPTERS=${VLLM_CHUNK_ADAPTERS:-16}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.82}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-1024}
VLLM_DTYPE=${VLLM_DTYPE:-bfloat16}
VLLM_ADAPTER_DTYPE=${VLLM_ADAPTER_DTYPE:-bfloat16}

CONFIRM_KS=${CONFIRM_KS:-1,2,4,8}
CONFIRM_MAX_K=${CONFIRM_MAX_K:-8}
CONFIRM_MAX_DENSE_REGRET=${CONFIRM_MAX_DENSE_REGRET:-0.0}
CONFIRM_MIN_FULL_SPEEDUP=${CONFIRM_MIN_FULL_SPEEDUP:-1.0}
PROPOSAL_SCORE_COL=${PROPOSAL_SCORE_COL:-selection_score}
SHORTLIST_POLICY=${SHORTLIST_POLICY:-}
CONFIRM_FAMILY_STATE_FILE=${CONFIRM_FAMILY_STATE_FILE:-}
SHORTLIST_REPORT_ARGS=()

export PYTHONUNBUFFERED=1
export VLLM_USAGE_STATS_ENABLED=${VLLM_USAGE_STATS_ENABLED:-0}
export VLLM_ENABLE_V1_MULTIPROCESSING=${VLLM_ENABLE_V1_MULTIPROCESSING:-0}
export XDG_CONFIG_HOME=${XDG_CONFIG_HOME:-/tmp/randopt-xdg-config}
mkdir -p "$OUT_ROOT" "$XDG_CONFIG_HOME"

extra_search_args=()
if [[ -n "$SIGMA_VALUES" ]]; then
  extra_search_args+=(--sigma-values "$SIGMA_VALUES")
fi
if [[ -n "$ENSEMBLE_KS" ]]; then
  extra_search_args+=(--ensemble-ks "$ENSEMBLE_KS")
fi
extra_confirm_args=("${extra_search_args[@]}")
if [[ -n "$CONFIRM_FAMILY_STATE_FILE" ]]; then
  extra_confirm_args+=(--family-state-file "$CONFIRM_FAMILY_STATE_FILE")
fi

extra_vllm_args=()
if [[ -n "$SIGMA_VALUES" ]]; then
  extra_vllm_args+=(--sigma-values "$SIGMA_VALUES")
fi
if [[ "$VLLM_REQUIRE_ALL_PROMPT_VARIANTS_VALID" == "1" ]]; then
  extra_vllm_args+=(--require-all-prompt-variants-valid)
fi

if [[ ! -f "$DATA" ]]; then
  "$PYTHON" -m randopt_lora_lab.make_countdown_data \
    --out "$DATA" \
    --count 1200 \
    --seed 20260507
fi

if [[ "$RUN_DENSE" == "1" ]]; then
  "$PYTHON" -m randopt_lora_lab.experiments search \
    --out "$OUT_ROOT/dense" \
    --model "$MODEL" \
    --data "$DATA" \
    --perturbation-backend dense \
    --family dense_gaussian \
    --population "$POPULATION" \
    --prompts "$PROMPTS" \
    --holdout-prompts "$HOLDOUT_PROMPTS" \
    --promote "$SHORTLIST_K" \
    --rank "$RANK" \
    --sigma "$SIGMA" \
    --seed "$SEED" \
    --targets "$TARGETS" \
    --batch-size "$HF_BATCH_SIZE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --stop-at-answer \
    "${extra_search_args[@]}"
  "$PYTHON" -m randopt_lora_lab.result_validity \
    --run "$OUT_ROOT/dense" \
    --out "$OUT_ROOT/dense/validity"
fi

if [[ "$RUN_VLLM" == "1" ]]; then
  "$PYTHON" -m randopt_lora_lab.vllm_lora_search \
    --out "$OUT_ROOT/vllm" \
    --model "$MODEL" \
    --data "$DATA" \
    --family "$FAMILY" \
    --population "$POPULATION" \
    --prompts "$PROMPTS" \
    --holdout-prompts "$VLLM_HOLDOUT_PROMPTS" \
    --promote 0 \
    --rank "$RANK" \
    --sigma "$SIGMA" \
    --seed "$SEED" \
    --targets "$TARGETS" \
    --prompt-input "$VLLM_PROMPT_INPUT" \
    --prompt-variants "$VLLM_PROMPT_VARIANTS" \
    --score-mode "$VLLM_SCORE_MODE" \
    --min-selection-prompt-variants "$VLLM_MIN_SELECTION_PROMPT_VARIANTS" \
    --max-base-malformed-for-selection "$VLLM_MAX_BASE_MALFORMED" \
    --max-base-cap-hit-for-selection "$VLLM_MAX_BASE_CAP_HIT" \
    --malformed-penalty "$VLLM_MALFORMED_PENALTY" \
    --cap-hit-penalty "$VLLM_CAP_HIT_PENALTY" \
    --max-loras "$VLLM_MAX_LORAS" \
    --chunk-adapters "$VLLM_CHUNK_ADAPTERS" \
    --max-cpu-loras 4096 \
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
    --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --dtype "$VLLM_DTYPE" \
    --adapter-dtype "$VLLM_ADAPTER_DTYPE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --stop-at-answer \
    --keep-adapters \
    "${extra_vllm_args[@]}"
fi

if [[ -n "$SHORTLIST_POLICY" ]]; then
  "$PYTHON" -m randopt_lora_lab.selector_union_audit shortlist \
    --run "$OUT_ROOT" \
    --out "$OUT_ROOT/shortlist_top${SHORTLIST_K}.jsonl" \
    --policy "$SHORTLIST_POLICY" \
    --k "$SHORTLIST_K"
  SHORTLIST_REPORT_ARGS+=(--candidate-file "$OUT_ROOT/shortlist_top${SHORTLIST_K}.jsonl")
else
  "$PYTHON" -m randopt_lora_lab.shortlist_from_run \
    --run "$OUT_ROOT/vllm" \
    --out "$OUT_ROOT/shortlist_top${SHORTLIST_K}.jsonl" \
    --k "$SHORTLIST_K" \
    --score-col "$PROPOSAL_SCORE_COL"
fi

if [[ "$RUN_CONFIRM" == "1" ]]; then
  if [[ -z "$CONFIRM_FAMILY_STATE_FILE" && -f "$OUT_ROOT/vllm/family_state.pt" ]]; then
    extra_confirm_args+=(--family-state-file "$OUT_ROOT/vllm/family_state.pt")
  fi
  "$PYTHON" -m randopt_lora_lab.experiments search \
    --out "$OUT_ROOT/confirmed" \
    --model "$MODEL" \
    --data "$DATA" \
    --perturbation-backend lora \
    --family "$FAMILY" \
    --candidate-file "$OUT_ROOT/shortlist_top${SHORTLIST_K}.jsonl" \
    --population "$SHORTLIST_K" \
    --prompts "$PROMPTS" \
    --holdout-prompts "$HOLDOUT_PROMPTS" \
    --promote "$SHORTLIST_K" \
    --rank "$RANK" \
    --sigma "$SIGMA" \
    --seed "$SEED" \
    --targets "$TARGETS" \
    --batch-size "$HF_BATCH_SIZE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --stop-at-answer \
    "${extra_confirm_args[@]}"
  "$PYTHON" -m randopt_lora_lab.result_validity \
    --run "$OUT_ROOT/confirmed" \
    --out "$OUT_ROOT/confirmed/validity"
fi

if [[ "$RUN_REPORT" == "1" ]]; then
  "$PYTHON" -m randopt_lora_lab.shortlist_dense_confirmation \
    --dense "$OUT_ROOT/dense" \
    --confirmed "$OUT_ROOT/confirmed" \
    --proposal "$OUT_ROOT/vllm" \
    --out "$OUT_ROOT/shortlist_dense_confirmation" \
    --ks "$CONFIRM_KS" \
    --proposal-score-col "$PROPOSAL_SCORE_COL" \
    "${SHORTLIST_REPORT_ARGS[@]}" \
    --max-confirm-k "$CONFIRM_MAX_K" \
    --max-dense-regret "$CONFIRM_MAX_DENSE_REGRET" \
    --min-full-without-dense-load-speedup "$CONFIRM_MIN_FULL_SPEEDUP"
fi

if [[ "$RUN_PROVENANCE_AUDIT" == "1" ]]; then
  "$PYTHON" -m randopt_lora_lab.family_state_provenance_audit \
    --root "$OUT_ROOT" \
    --out "$OUT_ROOT/family_state_provenance_audit"
fi
