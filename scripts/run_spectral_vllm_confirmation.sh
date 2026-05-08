#!/usr/bin/env bash
set -euo pipefail

DATA=${DATA:-data/countdown_generated_1200_seed20260507.json}
MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct}
OUT_ROOT=${OUT_ROOT:-results/spectral_vllm_confirmation_rank32_c1p5_p64}
PYTHON=${PYTHON:-python}

FAMILY=${FAMILY:-spectral_projected_gaussian_rank_r_c1p5}
CONTROL_FAMILY=${CONTROL_FAMILY:-factor_gaussian_lora}
POPULATION=${POPULATION:-64}
PROMPTS=${PROMPTS:-64}
HOLDOUT_PROMPTS=${HOLDOUT_PROMPTS:-256}
VLLM_HOLDOUT_PROMPTS=${VLLM_HOLDOUT_PROMPTS:-8}
PROMOTE=${PROMOTE:-16}
VLLM_PROMOTE=${VLLM_PROMOTE:-0}
RANK=${RANK:-32}
SIGMA=${SIGMA:-0.001}
SIGMA_VALUES=${SIGMA_VALUES:-0.0005,0.001,0.002}
SEED=${SEED:-20260507}
TARGETS=${TARGETS:-q_proj,v_proj}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
HF_BATCH_SIZE=${HF_BATCH_SIZE:-32}
ENSEMBLE_KS=${ENSEMBLE_KS:-1,4,8}

RUN_DENSE=${RUN_DENSE:-1}
RUN_CONTROL=${RUN_CONTROL:-1}
RUN_SPECTRAL=${RUN_SPECTRAL:-1}
RUN_VLLM=${RUN_VLLM:-1}
RUN_VLLM_FIRST=${RUN_VLLM_FIRST:-0}
RUN_CONFIRMATION=${RUN_CONFIRMATION:-1}
RUN_DENSE_REFERENCE_CONFIRMATION=${RUN_DENSE_REFERENCE_CONFIRMATION:-1}
RUN_VALIDITY=${RUN_VALIDITY:-1}
RUN_PARITY=${RUN_PARITY:-1}

VLLM_PROMPT_INPUT=${VLLM_PROMPT_INPUT:-token_ids}
VLLM_PROMPT_VARIANTS=${VLLM_PROMPT_VARIANTS:-default,reordered,xml}
VLLM_SCORE_MODE=${VLLM_SCORE_MODE:-robust_mean}
VLLM_MIN_SELECTION_PROMPT_VARIANTS=${VLLM_MIN_SELECTION_PROMPT_VARIANTS:-2}
VLLM_MAX_BASE_MALFORMED=${VLLM_MAX_BASE_MALFORMED:-0.05}
VLLM_MAX_BASE_CAP_HIT=${VLLM_MAX_BASE_CAP_HIT:-0.05}
VLLM_MALFORMED_PENALTY=${VLLM_MALFORMED_PENALTY:-1.0}
VLLM_CAP_HIT_PENALTY=${VLLM_CAP_HIT_PENALTY:-1.0}
VLLM_MAX_LORAS=${VLLM_MAX_LORAS:-16}
VLLM_CHUNK_ADAPTERS=${VLLM_CHUNK_ADAPTERS:-16}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.82}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-1024}
VLLM_MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS:-0}
VLLM_DTYPE=${VLLM_DTYPE:-bfloat16}
VLLM_ADAPTER_DTYPE=${VLLM_ADAPTER_DTYPE:-bfloat16}

CONFIRM_KS=${CONFIRM_KS:-1,2,4,8,16,32}
CONFIRM_MAX_K=${CONFIRM_MAX_K:-16}
CONFIRM_MIN_EVAL_ONLY_SPEEDUP=${CONFIRM_MIN_EVAL_ONLY_SPEEDUP:-1.0}
CONFIRM_MIN_FULL_WITHOUT_LOAD_SPEEDUP=${CONFIRM_MIN_FULL_WITHOUT_LOAD_SPEEDUP:-1.0}
CONFIRM_MAX_REGRET=${CONFIRM_MAX_REGRET:-0.0}
PROPOSAL_SCORE_COL=${PROPOSAL_SCORE_COL:-selection_score}
DENSE_REF_MAX_K=${DENSE_REF_MAX_K:-8}
DENSE_REF_MAX_REGRET=${DENSE_REF_MAX_REGRET:-0.0}
DENSE_REF_MIN_FULL_WITHOUT_LOAD_SPEEDUP=${DENSE_REF_MIN_FULL_WITHOUT_LOAD_SPEEDUP:-1.0}

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

extra_vllm_args=()
if [[ -n "$SIGMA_VALUES" ]]; then
  extra_vllm_args+=(--sigma-values "$SIGMA_VALUES")
fi
if [[ "$VLLM_MAX_NUM_BATCHED_TOKENS" != "0" ]]; then
  extra_vllm_args+=(--max-num-batched-tokens "$VLLM_MAX_NUM_BATCHED_TOKENS")
fi

run_peft_search() {
  local name="$1"
  local backend="$2"
  local family="$3"
  local out_dir="$OUT_ROOT/$name"

  "$PYTHON" -m randopt_lora_lab.experiments search \
    --out "$out_dir" \
    --model "$MODEL" \
    --data "$DATA" \
    --perturbation-backend "$backend" \
    --family "$family" \
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
    "${extra_search_args[@]}"

  if [[ "$RUN_VALIDITY" == "1" ]]; then
    "$PYTHON" -m randopt_lora_lab.result_validity \
      --run "$out_dir" \
      --out "$out_dir/validity"
  fi
}

run_vllm_search() {
  "$PYTHON" -m randopt_lora_lab.vllm_lora_search \
    --out "$OUT_ROOT/vllm_spectral" \
    --model "$MODEL" \
    --data "$DATA" \
    --family "$FAMILY" \
    --population "$POPULATION" \
    --prompts "$PROMPTS" \
    --holdout-prompts "$VLLM_HOLDOUT_PROMPTS" \
    --promote "$VLLM_PROMOTE" \
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
}

if [[ ! -f "$DATA" ]]; then
  "$PYTHON" -m randopt_lora_lab.make_countdown_data \
    --out "$DATA" \
    --count 1200 \
    --seed 20260507
fi

if [[ "$RUN_VLLM" == "1" && "$RUN_VLLM_FIRST" == "1" ]]; then
  run_vllm_search
fi

if [[ "$RUN_DENSE" == "1" ]]; then
  run_peft_search dense dense dense_gaussian
fi

if [[ "$RUN_CONTROL" == "1" ]]; then
  run_peft_search control lora "$CONTROL_FAMILY"
fi

if [[ "$RUN_SPECTRAL" == "1" ]]; then
  run_peft_search spectral lora "$FAMILY"
fi

if [[ "$RUN_PARITY" == "1" ]]; then
  extra_candidates=()
  if [[ -f "$OUT_ROOT/control/summary.json" ]]; then
    extra_candidates+=(--candidate "control=$OUT_ROOT/control")
  fi
  "$PYTHON" -m randopt_lora_lab.parity_report \
    --dense "$OUT_ROOT/dense" \
    --lora "$OUT_ROOT/spectral" \
    "${extra_candidates[@]}" \
    --out "$OUT_ROOT/parity"
fi

if [[ "$RUN_VLLM" == "1" && "$RUN_VLLM_FIRST" != "1" ]]; then
  run_vllm_search
fi

if [[ "$RUN_CONFIRMATION" == "1" ]]; then
  "$PYTHON" -m randopt_lora_lab.confirmation_economics \
    --trusted "$OUT_ROOT/spectral" \
    --proposal "$OUT_ROOT/vllm_spectral" \
    --out "$OUT_ROOT/confirmation" \
    --ks "$CONFIRM_KS" \
    --proposal-score-col "$PROPOSAL_SCORE_COL" \
    --max-confirm-k "$CONFIRM_MAX_K" \
    --min-eval-only-speedup "$CONFIRM_MIN_EVAL_ONLY_SPEEDUP" \
    --min-full-without-load-speedup "$CONFIRM_MIN_FULL_WITHOUT_LOAD_SPEEDUP" \
    --max-regret "$CONFIRM_MAX_REGRET"
fi

if [[ "$RUN_DENSE_REFERENCE_CONFIRMATION" == "1" ]]; then
  "$PYTHON" -m randopt_lora_lab.dense_reference_confirmation \
    --dense "$OUT_ROOT/dense" \
    --trusted-spectral "$OUT_ROOT/spectral" \
    --proposal "$OUT_ROOT/vllm_spectral" \
    --out "$OUT_ROOT/dense_reference_confirmation" \
    --ks "$CONFIRM_KS" \
    --proposal-score-col "$PROPOSAL_SCORE_COL" \
    --max-confirm-k "$DENSE_REF_MAX_K" \
    --max-dense-regret "$DENSE_REF_MAX_REGRET" \
    --min-full-without-dense-load-speedup "$DENSE_REF_MIN_FULL_WITHOUT_LOAD_SPEEDUP"
fi
