#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}
OUT_ROOT=${OUT_ROOT:-results/spectral_vllm_multirun_rank32_c1p5}
RUN_SEEDS=${RUN_SEEDS:-20260507,20260508}
PARITY_ARM=${PARITY_ARM:-lora}
MIN_RUNS=${MIN_RUNS:-2}
MIN_PROMPT_VARIANTS=${MIN_PROMPT_VARIANTS:-2}
MAX_ZERO_REGRET_K=${MAX_ZERO_REGRET_K:-8}
MIN_FULL_WITHOUT_LOAD_SPEEDUP=${MIN_FULL_WITHOUT_LOAD_SPEEDUP:-1.0}

mkdir -p "$OUT_ROOT"

IFS=',' read -r -a seed_values <<< "$RUN_SEEDS"
run_args=()
for seed in "${seed_values[@]}"; do
  seed=${seed//[[:space:]]/}
  [[ -n "$seed" ]] || continue
  run_dir="$OUT_ROOT/seed${seed}"
  run_args+=(--run "$run_dir")
  OUT_ROOT="$run_dir" \
  SEED="$seed" \
  RUN_VLLM_FIRST="${RUN_VLLM_FIRST:-1}" \
  scripts/run_spectral_vllm_confirmation.sh
done

"$PYTHON" -m randopt_lora_lab.multirun_gate \
  "${run_args[@]}" \
  --parity-arm "$PARITY_ARM" \
  --min-runs "$MIN_RUNS" \
  --min-prompt-variants "$MIN_PROMPT_VARIANTS" \
  --max-zero-regret-k "$MAX_ZERO_REGRET_K" \
  --min-full-without-load-speedup "$MIN_FULL_WITHOUT_LOAD_SPEEDUP" \
  --out "$OUT_ROOT/gate"
