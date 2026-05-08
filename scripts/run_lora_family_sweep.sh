#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run a matched dense/factor/sparse LoRA family sweep.

Configure with environment variables, for example:

  OUT_ROOT=results/lora_family_sweep_rank32_p64 \
  FAMILIES=factor_gaussian_lora,sparse_low_rank_lora_d0p25,sparse_low_rank_lora_d0p125 \
  PROMPT_VARIANTS=default,reordered \
  POPULATION=64 \
  scripts/run_lora_family_sweep.sh

EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ "$#" -ne 0 ]]; then
  usage >&2
  echo "unexpected positional arguments: $*" >&2
  exit 2
fi

DATA=${DATA:-data/countdown_generated_1200_seed20260507.json}
MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct}
OUT_ROOT=${OUT_ROOT:-results/lora_family_sweep_rank32_p64}
PYTHON=${PYTHON:-python}

FAMILIES=${FAMILIES:-factor_gaussian_lora,sparse_low_rank_lora_d0p25,sparse_low_rank_lora_d0p125}
BASELINE_FAMILY=${BASELINE_FAMILY:-factor_gaussian_lora}
BASELINE_ARM=${BASELINE_ARM:-lora}
PROMPT_VARIANTS=${PROMPT_VARIANTS:-default,reordered}
POPULATION=${POPULATION:-64}
PROMPTS=${PROMPTS:-64}
HOLDOUT_PROMPTS=${HOLDOUT_PROMPTS:-256}
PROMOTE=${PROMOTE:-16}
RANK=${RANK:-32}
SIGMA=${SIGMA:-0.001}
SIGMA_VALUES=${SIGMA_VALUES:-0.0005,0.001,0.002}
SEED=${SEED:-20260507}
TARGETS=${TARGETS:-q_proj,v_proj}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
HF_BATCH_SIZE=${HF_BATCH_SIZE:-32}
ENSEMBLE_KS=${ENSEMBLE_KS:-1,4,8,16}

MIN_VARIANTS=${MIN_VARIANTS:-2}
MIN_IMPROVEMENT_EXAMPLES=${MIN_IMPROVEMENT_EXAMPLES:-2}
MAX_CAP_HIT_DELTA=${MAX_CAP_HIT_DELTA:-0.02}
MAX_MALFORMED_DELTA=${MAX_MALFORMED_DELTA:-0.02}
MIN_SPEED_RATIO_OVER_DENSE=${MIN_SPEED_RATIO_OVER_DENSE:-1.0}

export PYTHONUNBUFFERED=1
mkdir -p "$OUT_ROOT"

family_slug() {
  case "$1" in
    factor_gaussian_lora)
      echo "factor"
      ;;
    sparse_low_rank_lora_d*)
      echo "sparse_${1#sparse_low_rank_lora_}"
      ;;
    *)
      echo "$1" | tr -c '[:alnum:]_' '_'
      ;;
  esac
}

extra_search_args=()
if [[ -n "$SIGMA_VALUES" ]]; then
  extra_search_args+=(--sigma-values "$SIGMA_VALUES")
fi
if [[ -n "$ENSEMBLE_KS" ]]; then
  extra_search_args+=(--ensemble-ks "$ENSEMBLE_KS")
fi

run_search() {
  local out_dir="$1"
  local backend="$2"
  local family="$3"
  local variant="$4"

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
    --prompt-variant "$variant" \
    --stop-at-answer \
    "${extra_search_args[@]}"

  "$PYTHON" -m randopt_lora_lab.result_validity \
    --run "$out_dir" \
    --out "$out_dir/validity"
}

if [[ ! -f "$DATA" ]]; then
  "$PYTHON" -m randopt_lora_lab.make_countdown_data \
    --out "$DATA" \
    --count 1200 \
    --seed 20260507
fi

IFS=',' read -r -a family_values <<< "$FAMILIES"
IFS=',' read -r -a variant_values <<< "$PROMPT_VARIANTS"
variant_args=()

for variant in "${variant_values[@]}"; do
  variant="$(echo "$variant" | tr -d '[:space:]')"
  [[ -n "$variant" ]] || continue
  variant_dir="$OUT_ROOT/$variant"
  mkdir -p "$variant_dir"
  variant_args+=(--variant-root "$variant=$variant_dir")

  run_search "$variant_dir/dense" dense dense_gaussian "$variant"

  candidate_args=()
  lora_dir=""
  for family in "${family_values[@]}"; do
    family="$(echo "$family" | tr -d '[:space:]')"
    [[ -n "$family" ]] || continue
    arm="$(family_slug "$family")"
    out_dir="$variant_dir/$arm"
    run_search "$out_dir" lora "$family" "$variant"
    if [[ "$family" == "$BASELINE_FAMILY" ]]; then
      lora_dir="$out_dir"
    else
      candidate_args+=(--candidate "$arm=$out_dir")
    fi
  done

  if [[ -z "$lora_dir" ]]; then
    echo "BASELINE_FAMILY=$BASELINE_FAMILY was not present in FAMILIES=$FAMILIES" >&2
    exit 1
  fi

  "$PYTHON" -m randopt_lora_lab.parity_report \
    --dense "$variant_dir/dense" \
    --lora "$lora_dir" \
    "${candidate_args[@]}" \
    --out "$variant_dir/parity"
done

"$PYTHON" -m randopt_lora_lab.family_sweep_report \
  "${variant_args[@]}" \
  --out "$OUT_ROOT/report" \
  --baseline-arm "$BASELINE_ARM" \
  --min-variants "$MIN_VARIANTS" \
  --min-improvement-examples "$MIN_IMPROVEMENT_EXAMPLES" \
  --max-cap-hit-delta "$MAX_CAP_HIT_DELTA" \
  --max-malformed-delta "$MAX_MALFORMED_DELTA" \
  --min-speed-ratio-over-dense "$MIN_SPEED_RATIO_OVER_DENSE"
