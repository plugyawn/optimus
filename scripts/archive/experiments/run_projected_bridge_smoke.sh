#!/usr/bin/env bash
set -euo pipefail

BASE_OUT=${BASE_OUT:-results/projected_bridge_smoke_p16}
DATA=${DATA:-data/countdown_generated_1200_seed20260507.json}
PYTHON=${PYTHON:-python}
RANKS=${RANKS:-8,32,64}
POPULATION=${POPULATION:-16}
PROMPTS=${PROMPTS:-64}
HOLDOUT_PROMPTS=${HOLDOUT_PROMPTS:-256}
SIGMA_VALUES=${SIGMA_VALUES:-0.0005,0.001,0.002}
SIGMA=${SIGMA:-0.001}
PROMOTE=${PROMOTE:-4}
ENSEMBLE_KS=${ENSEMBLE_KS:-1,4}
ENSEMBLE_RATIOS=${ENSEMBLE_RATIOS:-}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
BATCH_SIZE=${BATCH_SIZE:-32}
REUSE_DENSE=${REUSE_DENSE:-1}

if [[ ! -f "$DATA" ]]; then
  "$PYTHON" -m randopt_lora_lab.make_countdown_data \
    --out "$DATA" \
    --count 1200 \
    --seed 20260507
fi

BASE_OUT="$BASE_OUT" \
DATA="$DATA" \
PYTHON="$PYTHON" \
RANKS="$RANKS" \
REUSE_DENSE="$REUSE_DENSE" \
POPULATION="$POPULATION" \
PROMPTS="$PROMPTS" \
HOLDOUT_PROMPTS="$HOLDOUT_PROMPTS" \
SIGMA_VALUES="$SIGMA_VALUES" \
SIGMA="$SIGMA" \
PROMOTE="$PROMOTE" \
ENSEMBLE_KS="$ENSEMBLE_KS" \
ENSEMBLE_RATIOS="$ENSEMBLE_RATIOS" \
MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
BATCH_SIZE="$BATCH_SIZE" \
INCLUDE_PROJECTED=1 \
scripts/run_gaussian_parity_rank_sweep.sh

IFS=',' read -r -a rank_values <<< "$RANKS"
for rank in "${rank_values[@]}"; do
  rank="$(echo "$rank" | tr -d '[:space:]')"
  [[ -z "$rank" ]] && continue
  for arm in dense lora projected; do
    run_dir="$BASE_OUT/rank${rank}/${arm}"
    [[ -f "$run_dir/summary.json" ]] || continue
    "$PYTHON" -m randopt_lora_lab.result_validity \
      --run "$run_dir" \
      --out "$run_dir/validity"
  done
done
