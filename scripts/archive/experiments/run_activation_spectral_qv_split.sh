#!/usr/bin/env bash
set -euo pipefail

OUT_ROOT=${OUT_ROOT:-results/activation_spectral_qv_split_p32_a100}
DATA=${DATA:-data/countdown_generated_1200_seed20260507.json}
MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct}
PYTHON=${PYTHON:-python}
TARGET_SETS=${TARGET_SETS:-q_proj,v_proj,q_proj+v_proj}
FAMILIES=${FAMILIES:-activation_spectral_lora,activation_spectral_lora_c2}
POPULATION=${POPULATION:-32}
PROMPTS=${PROMPTS:-64}
HOLDOUT_PROMPTS=${HOLDOUT_PROMPTS:-128}
RANK=${RANK:-32}
SIGMA=${SIGMA:-0.001}
SIGMA_VALUES=${SIGMA_VALUES:-0.0005,0.001,0.002}
SEED=${SEED:-20260507}
PROMOTE=${PROMOTE:-8}
ENSEMBLE_KS=${ENSEMBLE_KS:-1,4,8}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
BATCH_SIZE=${BATCH_SIZE:-16}
RUN_VALIDITY=${RUN_VALIDITY:-1}

if [[ ! -f "$DATA" ]]; then
  "$PYTHON" -m randopt_lora_lab.make_countdown_data \
    --out "$DATA" \
    --count 1200 \
    --seed 20260507
fi

extra_search_args=()
if [[ -n "$SIGMA_VALUES" ]]; then
  extra_search_args+=(--sigma-values "$SIGMA_VALUES")
fi
if [[ -n "$ENSEMBLE_KS" ]]; then
  extra_search_args+=(--ensemble-ks "$ENSEMBLE_KS")
fi

target_slug() {
  echo "$1" | tr ',' '_' | tr '+' '_'
}

normalize_targets() {
  echo "$1" | tr '+' ','
}

IFS=',' read -r -a target_values <<< "$TARGET_SETS"
IFS=',' read -r -a family_values <<< "$FAMILIES"

mkdir -p "$OUT_ROOT"

for target_set in "${target_values[@]}"; do
  target_set="$(echo "$target_set" | tr -d '[:space:]')"
  [[ -n "$target_set" ]] || continue
  targets="$(normalize_targets "$target_set")"
  tslug="$(target_slug "$target_set")"
  for family in "${family_values[@]}"; do
    family="$(echo "$family" | tr -d '[:space:]')"
    [[ -n "$family" ]] || continue
    out_dir="$OUT_ROOT/${tslug}_${family}"
    "$PYTHON" -m randopt_lora_lab.experiments search \
      --out "$out_dir" \
      --model "$MODEL" \
      --data "$DATA" \
      --perturbation-backend lora \
      --family "$family" \
      --population "$POPULATION" \
      --prompts "$PROMPTS" \
      --holdout-prompts "$HOLDOUT_PROMPTS" \
      --rank "$RANK" \
      --sigma "$SIGMA" \
      --seed "$SEED" \
      --targets "$targets" \
      --promote "$PROMOTE" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --batch-size "$BATCH_SIZE" \
      --stop-at-answer \
      "${extra_search_args[@]}"

    if [[ "$RUN_VALIDITY" == "1" ]]; then
      "$PYTHON" -m randopt_lora_lab.result_validity \
        --run "$out_dir" \
        --out "$out_dir/validity"
    fi
  done
done
