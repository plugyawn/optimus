#!/usr/bin/env bash
set -euo pipefail

DATA=${DATA:-data/countdown_generated_1200_seed20260507.json}
MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct}
OUT=${OUT:-results/gaussian_parity_baseline}
PYTHON=${PYTHON:-python}
POPULATION=${POPULATION:-64}
PROMPTS=${PROMPTS:-64}
HOLDOUT_PROMPTS=${HOLDOUT_PROMPTS:-256}
SIGMA=${SIGMA:-0.01}
SIGMA_VALUES=${SIGMA_VALUES:-}
SEED=${SEED:-20260507}
RANK=${RANK:-8}
TARGETS=${TARGETS:-q_proj,v_proj}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-32}
BATCH_SIZE=${BATCH_SIZE:-16}
PROMPT_VARIANT=${PROMPT_VARIANT:-default}
USE_CHAT_TEMPLATE=${USE_CHAT_TEMPLATE:-0}
PROMOTE=${PROMOTE:-4}
ENSEMBLE_KS=${ENSEMBLE_KS:-}
ENSEMBLE_RATIOS=${ENSEMBLE_RATIOS:-}
DENSE_SNAPSHOT_DEVICE=${DENSE_SNAPSHOT_DEVICE:-model}
DENSE_NOISE_MODE=${DENSE_NOISE_MODE:-canonical}
INCLUDE_PROJECTED=${INCLUDE_PROJECTED:-1}
DENSE_REF_DIR=${DENSE_REF_DIR:-}
RUN_DENSE=${RUN_DENSE:-1}

extra_search_args=()
if [[ -n "$SIGMA_VALUES" ]]; then
  extra_search_args+=(--sigma-values "$SIGMA_VALUES")
fi
if [[ -n "$ENSEMBLE_KS" ]]; then
  extra_search_args+=(--ensemble-ks "$ENSEMBLE_KS")
fi
if [[ -n "$ENSEMBLE_RATIOS" ]]; then
  extra_search_args+=(--ensemble-ratios "$ENSEMBLE_RATIOS")
fi
if [[ "$PROMPT_VARIANT" != "default" ]]; then
  extra_search_args+=(--prompt-variant "$PROMPT_VARIANT")
fi
if [[ "$USE_CHAT_TEMPLATE" == "1" ]]; then
  extra_search_args+=(--use-chat-template)
fi
if [[ "$DENSE_NOISE_MODE" != "canonical" ]]; then
  extra_search_args+=(--dense-noise-mode "$DENSE_NOISE_MODE")
fi

if [[ ! -f "$DATA" ]]; then
  "$PYTHON" -m randopt_lora_lab.make_countdown_data \
    --out "$DATA" \
    --count 1200 \
    --seed 20260507
fi

if [[ -n "$DENSE_REF_DIR" ]]; then
  rm -rf "$OUT/dense"
  mkdir -p "$OUT"
  cp -a "$DENSE_REF_DIR" "$OUT/dense"
elif [[ "$RUN_DENSE" == "1" ]]; then
  "$PYTHON" -m randopt_lora_lab.experiments search \
    --out "$OUT/dense" \
    --model "$MODEL" \
    --data "$DATA" \
    --perturbation-backend dense \
    --family dense_gaussian \
    --population "$POPULATION" \
    --prompts "$PROMPTS" \
    --holdout-prompts "$HOLDOUT_PROMPTS" \
    --rank "$RANK" \
    --sigma "$SIGMA" \
    --targets "$TARGETS" \
    --promote "$PROMOTE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --batch-size "$BATCH_SIZE" \
    --dense-snapshot-device "$DENSE_SNAPSHOT_DEVICE" \
    --seed "$SEED" \
    --stop-at-answer \
    "${extra_search_args[@]}"
elif [[ ! -f "$OUT/dense/summary.json" ]]; then
  echo "RUN_DENSE=0 requires an existing $OUT/dense/summary.json or DENSE_REF_DIR" >&2
  exit 1
fi

"$PYTHON" -m randopt_lora_lab.experiments search \
  --out "$OUT/lora" \
  --model "$MODEL" \
  --data "$DATA" \
  --perturbation-backend lora \
  --family factor_gaussian_lora \
  --population "$POPULATION" \
  --prompts "$PROMPTS" \
  --holdout-prompts "$HOLDOUT_PROMPTS" \
  --rank "$RANK" \
  --sigma "$SIGMA" \
  --targets "$TARGETS" \
  --promote "$PROMOTE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --batch-size "$BATCH_SIZE" \
  --seed "$SEED" \
  --stop-at-answer \
  "${extra_search_args[@]}"

extra_candidates=()
if [[ "$INCLUDE_PROJECTED" == "1" ]]; then
  "$PYTHON" -m randopt_lora_lab.experiments search \
    --out "$OUT/projected" \
    --model "$MODEL" \
    --data "$DATA" \
    --perturbation-backend lora \
    --family projected_gaussian_rank_r \
    --population "$POPULATION" \
    --prompts "$PROMPTS" \
    --holdout-prompts "$HOLDOUT_PROMPTS" \
    --rank "$RANK" \
    --sigma "$SIGMA" \
    --targets "$TARGETS" \
    --promote "$PROMOTE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --batch-size "$BATCH_SIZE" \
    --seed "$SEED" \
    --stop-at-answer \
    "${extra_search_args[@]}"
  extra_candidates+=(--candidate "projected=$OUT/projected")
fi

"$PYTHON" -m randopt_lora_lab.parity_report \
  --dense "$OUT/dense" \
  --lora "$OUT/lora" \
  "${extra_candidates[@]}" \
  --out "$OUT/report"
