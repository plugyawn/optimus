#!/usr/bin/env bash
set -euo pipefail

BASE_OUT=${BASE_OUT:-results/gaussian_parity_rank_sweep}
RANKS=${RANKS:-8,32}
REUSE_DENSE=${REUSE_DENSE:-1}

IFS=',' read -r -a rank_values <<< "$RANKS"
first_dense_dir=""
for rank in "${rank_values[@]}"; do
  rank="$(echo "$rank" | tr -d '[:space:]')"
  if [[ -z "$rank" ]]; then
    continue
  fi
  if [[ "$REUSE_DENSE" == "1" && -n "$first_dense_dir" ]]; then
    OUT="$BASE_OUT/rank${rank}" RANK="$rank" DENSE_REF_DIR="$first_dense_dir" scripts/run_gaussian_parity_baseline.sh
  else
    OUT="$BASE_OUT/rank${rank}" RANK="$rank" scripts/run_gaussian_parity_baseline.sh
    if [[ -z "$first_dense_dir" ]]; then
      first_dense_dir="$BASE_OUT/rank${rank}/dense"
    fi
  fi
done
