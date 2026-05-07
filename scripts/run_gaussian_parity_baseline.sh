#!/usr/bin/env bash
set -euo pipefail

DATA=${DATA:-data/countdown_generated_1200_seed20260507.json}
MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct}
OUT=${OUT:-results/gaussian_parity_baseline}
POPULATION=${POPULATION:-64}
PROMPTS=${PROMPTS:-64}
HOLDOUT_PROMPTS=${HOLDOUT_PROMPTS:-256}
SIGMA=${SIGMA:-0.01}
SEED=${SEED:-20260507}

if [[ ! -f "$DATA" ]]; then
  python -m randopt_lora_lab.make_countdown_data \
    --out "$DATA" \
    --count 1200 \
    --seed 20260507
fi

python -m randopt_lora_lab.experiments search \
  --out "$OUT/dense" \
  --model "$MODEL" \
  --data "$DATA" \
  --perturbation-backend dense \
  --family dense_gaussian \
  --population "$POPULATION" \
  --prompts "$PROMPTS" \
  --holdout-prompts "$HOLDOUT_PROMPTS" \
  --sigma "$SIGMA" \
  --seed "$SEED" \
  --stop-at-answer

python -m randopt_lora_lab.experiments search \
  --out "$OUT/lora" \
  --model "$MODEL" \
  --data "$DATA" \
  --perturbation-backend lora \
  --family factor_gaussian_lora \
  --population "$POPULATION" \
  --prompts "$PROMPTS" \
  --holdout-prompts "$HOLDOUT_PROMPTS" \
  --sigma "$SIGMA" \
  --seed "$SEED" \
  --stop-at-answer

python -m randopt_lora_lab.experiments search \
  --out "$OUT/projected" \
  --model "$MODEL" \
  --data "$DATA" \
  --perturbation-backend lora \
  --family projected_gaussian_rank_r \
  --population "$POPULATION" \
  --prompts "$PROMPTS" \
  --holdout-prompts "$HOLDOUT_PROMPTS" \
  --sigma "$SIGMA" \
  --seed "$SEED" \
  --stop-at-answer

python -m randopt_lora_lab.parity_report \
  --dense "$OUT/dense" \
  --lora "$OUT/lora" \
  --candidate "projected=$OUT/projected" \
  --out "$OUT/report"
