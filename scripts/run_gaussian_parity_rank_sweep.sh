#!/usr/bin/env bash
set -euo pipefail

BASE_OUT=${BASE_OUT:-results/gaussian_parity_rank_sweep}
RANKS=${RANKS:-8,32}

IFS=',' read -r -a rank_values <<< "$RANKS"
for rank in "${rank_values[@]}"; do
  rank="$(echo "$rank" | tr -d '[:space:]')"
  if [[ -z "$rank" ]]; then
    continue
  fi
  OUT="$BASE_OUT/rank${rank}" RANK="$rank" scripts/run_gaussian_parity_baseline.sh
done

