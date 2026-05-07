#!/usr/bin/env bash
set -euo pipefail

BASE_OUT=${BASE_OUT:-results/spectral_calibration_smoke_p16}
REF_ROOT=${REF_ROOT:-results/projected_bridge_smoke_p16_a100}
DATA=${DATA:-data/countdown_generated_1200_seed20260507.json}
MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct}
PYTHON=${PYTHON:-python}
RANKS=${RANKS:-32,64}
FAMILIES=${FAMILIES:-spectral_projected_gaussian_rank_r_c0p5,spectral_projected_gaussian_rank_r_c0p75,spectral_projected_gaussian_rank_r,spectral_projected_gaussian_rank_r_c1p25,spectral_projected_gaussian_rank_r_c1p5,spectral_projected_gaussian_rank_r_c2}
POPULATION=${POPULATION:-16}
PROMPTS=${PROMPTS:-64}
HOLDOUT_PROMPTS=${HOLDOUT_PROMPTS:-256}
SIGMA=${SIGMA:-0.001}
SIGMA_VALUES=${SIGMA_VALUES:-0.0005,0.001,0.002}
SEED=${SEED:-20260507}
TARGETS=${TARGETS:-q_proj,v_proj}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
BATCH_SIZE=${BATCH_SIZE:-32}
PROMOTE=${PROMOTE:-4}
ENSEMBLE_KS=${ENSEMBLE_KS:-1,4}
RUN_BASELINES=${RUN_BASELINES:-0}

extra_search_args=()
if [[ -n "$SIGMA_VALUES" ]]; then
  extra_search_args+=(--sigma-values "$SIGMA_VALUES")
fi
if [[ -n "$ENSEMBLE_KS" ]]; then
  extra_search_args+=(--ensemble-ks "$ENSEMBLE_KS")
fi

if [[ ! -f "$DATA" ]]; then
  "$PYTHON" -m randopt_lora_lab.make_countdown_data \
    --out "$DATA" \
    --count 1200 \
    --seed 20260507
fi

family_slug() {
  local family="$1"
  case "$family" in
    spectral_projected_gaussian_rank_r)
      echo "spectral_c1"
      ;;
    spectral_projected_gaussian_rank_r_c*)
      echo "spectral_${family#spectral_projected_gaussian_rank_r_}"
      ;;
    *)
      echo "$family"
      ;;
  esac
}

copy_or_run_baseline() {
  local rank="$1"
  local arm="$2"
  local family="$3"
  local backend="$4"
  local out_dir="$BASE_OUT/rank${rank}/${arm}"
  local ref_dir="$REF_ROOT/rank${rank}/${arm}"

  if [[ "$RUN_BASELINES" != "1" && -f "$ref_dir/summary.json" && -f "$ref_dir/candidate_summary.jsonl" ]]; then
    rm -rf "$out_dir"
    mkdir -p "$BASE_OUT/rank${rank}"
    cp -a "$ref_dir" "$out_dir"
    return
  fi

  "$PYTHON" -m randopt_lora_lab.experiments search \
    --out "$out_dir" \
    --model "$MODEL" \
    --data "$DATA" \
    --perturbation-backend "$backend" \
    --family "$family" \
    --population "$POPULATION" \
    --prompts "$PROMPTS" \
    --holdout-prompts "$HOLDOUT_PROMPTS" \
    --rank "$rank" \
    --sigma "$SIGMA" \
    --targets "$TARGETS" \
    --promote "$PROMOTE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --batch-size "$BATCH_SIZE" \
    --seed "$SEED" \
    --stop-at-answer \
    "${extra_search_args[@]}"
}

IFS=',' read -r -a rank_values <<< "$RANKS"
IFS=',' read -r -a family_values <<< "$FAMILIES"

for rank in "${rank_values[@]}"; do
  rank="$(echo "$rank" | tr -d '[:space:]')"
  [[ -z "$rank" ]] && continue

  copy_or_run_baseline "$rank" dense dense_gaussian dense
  copy_or_run_baseline "$rank" lora factor_gaussian_lora lora

  extra_candidates=()

  for family in "${family_values[@]}"; do
    family="$(echo "$family" | tr -d '[:space:]')"
    [[ -z "$family" ]] && continue
    slug="$(family_slug "$family")"
    run_dir="$BASE_OUT/rank${rank}/${slug}"
    "$PYTHON" -m randopt_lora_lab.experiments search \
      --out "$run_dir" \
      --model "$MODEL" \
      --data "$DATA" \
      --perturbation-backend lora \
      --family "$family" \
      --population "$POPULATION" \
      --prompts "$PROMPTS" \
      --holdout-prompts "$HOLDOUT_PROMPTS" \
      --rank "$rank" \
      --sigma "$SIGMA" \
      --targets "$TARGETS" \
      --promote "$PROMOTE" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --batch-size "$BATCH_SIZE" \
      --seed "$SEED" \
      --stop-at-answer \
      "${extra_search_args[@]}"
    "$PYTHON" -m randopt_lora_lab.result_validity \
      --run "$run_dir" \
      --out "$run_dir/validity"
    extra_candidates+=(--candidate "$slug=$run_dir")
  done

  "$PYTHON" -m randopt_lora_lab.parity_report \
    --dense "$BASE_OUT/rank${rank}/dense" \
    --lora "$BASE_OUT/rank${rank}/lora" \
    "${extra_candidates[@]}" \
    --out "$BASE_OUT/rank${rank}/report"
done

"$PYTHON" -m randopt_lora_lab.rank_sweep_report \
  --root "$BASE_OUT" \
  --out "$BASE_OUT/rank_sweep_summary"
