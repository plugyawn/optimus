#!/usr/bin/env bash
set -euo pipefail

DATA=${DATA:-data/countdown_generated_1200_seed20260507.json}
MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct}
PROMPTS=${PROMPTS:-64}
HOLDOUT_PROMPTS=${HOLDOUT_PROMPTS:-256}
POPULATION=${POPULATION:-128}
PROMOTE=${PROMOTE:-8}
BATCH_SIZE=${BATCH_SIZE:-16}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-32}

if [[ ! -f "$DATA" ]]; then
  python -m randopt_lora_lab.make_countdown_data --out "$DATA" --count 1200 --seed 20260507
fi

common=(
  --model "$MODEL"
  --data "$DATA"
  --prompts "$PROMPTS"
  --holdout-prompts "$HOLDOUT_PROMPTS"
  --population "$POPULATION"
  --promote "$PROMOTE"
  --batch-size "$BATCH_SIZE"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --stop-at-answer
  --antithetic
)

python -m randopt_lora_lab.experiments oracle \
  --out results/phase7_oracle \
  --model "$MODEL" \
  --data "$DATA" \
  --prompts "$PROMPTS" \
  --batch-size "$BATCH_SIZE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --stop-at-answer

python -m randopt_lora_lab.experiments search --out results/phase7_iso_s0p0075 "${common[@]}" --family isotropic --sigma 0.0075 --seed 2468
python -m randopt_lora_lab.experiments search --out results/phase7_iso_s0p01 "${common[@]}" --family isotropic --sigma 0.01 --seed 5678
python -m randopt_lora_lab.experiments search --out results/phase7_anzo_s0p015 "${common[@]}" --family anzo --sigma 0.015 --seed 2468
python -m randopt_lora_lab.experiments search --out results/phase7_anzo_s0p02 "${common[@]}" --family anzo --sigma 0.02 --seed 5678
python -m randopt_lora_lab.experiments search --out results/phase7_target_svd_s0p015 "${common[@]}" --family target_svd --sigma 0.015 --seed 2468
python -m randopt_lora_lab.experiments search --out results/phase7_random_ortho_s0p015 "${common[@]}" --family random_ortho --sigma 0.015 --seed 2468
python -m randopt_lora_lab.experiments search --out results/phase7_anzo_random_target_s0p015 "${common[@]}" --family anzo_random_target --sigma 0.015 --seed 2468

python -m randopt_lora_lab.report --root results --out results/report
