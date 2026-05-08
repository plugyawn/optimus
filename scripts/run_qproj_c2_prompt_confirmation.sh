#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}
SOURCE_RUN=${SOURCE_RUN:-results/activation_spectral_qv_split_p32_a100/q_proj_activation_spectral_lora_c2}
OUT_ROOT=${OUT_ROOT:-results/qproj_c2_prompt_confirmation}
DATA=${DATA:-data/countdown_generated_1200_seed20260507.json}
MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct}
TARGETS=${TARGETS:-q_proj}
BASE_RANK=${BASE_RANK:-32}
TOP_K=${TOP_K:-4}
PROMPTS=${PROMPTS:-64}
HOLDOUT_PROMPTS=${HOLDOUT_PROMPTS:-128}
SEED=${SEED:-20260507}
CAPS=${CAPS:-64,128,256}
PROMPT_VARIANTS=${PROMPT_VARIANTS:-default,reordered,xml}
BATCH_SIZE=${BATCH_SIZE:-16}
WEIGHT_MODE=${WEIGHT_MODE:-score}
RUN_AGGREGATE=${RUN_AGGREGATE:-0}
MIN_VALID_PROMPTS=${MIN_VALID_PROMPTS:-2}
MIN_LIFT=${MIN_LIFT:-0.015625}
MAX_CANDIDATE_MALFORMED=${MAX_CANDIDATE_MALFORMED:-0.05}
MAX_CANDIDATE_CAP_HIT=${MAX_CANDIDATE_CAP_HIT:-0.05}
MAX_MALFORMED_REGRESSION=${MAX_MALFORMED_REGRESSION:-0.05}
MAX_CAP_HIT_REGRESSION=${MAX_CAP_HIT_REGRESSION:-0.05}

if [[ ! -f "$DATA" ]]; then
  "$PYTHON" -m randopt_lora_lab.make_countdown_data \
    --out "$DATA" \
    --count 1200 \
    --seed 20260507
fi

mkdir -p "$OUT_ROOT"

cap_args=()
if [[ "$RUN_AGGREGATE" != "1" ]]; then
  cap_args+=(--skip-aggregate)
fi

"$PYTHON" -m randopt_lora_lab.cap_stability \
  --source-run "$SOURCE_RUN" \
  --out "$OUT_ROOT/cap_stability" \
  --model "$MODEL" \
  --data "$DATA" \
  --prompts "$PROMPTS" \
  --holdout-prompts "$HOLDOUT_PROMPTS" \
  --seed "$SEED" \
  --base-rank "$BASE_RANK" \
  --top-k "$TOP_K" \
  --weight-mode "$WEIGHT_MODE" \
  --targets "$TARGETS" \
  --max-new-tokens-grid "$CAPS" \
  --prompt-variants "$PROMPT_VARIANTS" \
  --batch-size "$BATCH_SIZE" \
  --stop-at-answer \
  "${cap_args[@]}"

"$PYTHON" -m randopt_lora_lab.prompt_ensemble_robustness \
  --summary "$OUT_ROOT/cap_stability/summary.json" \
  --out "$OUT_ROOT/ensemble_k${TOP_K}_strict" \
  --split holdout \
  --k "$TOP_K" \
  --strict-rows \
  --min-valid-prompts "$MIN_VALID_PROMPTS" \
  --min-lift "$MIN_LIFT" \
  --max-candidate-malformed "$MAX_CANDIDATE_MALFORMED" \
  --max-candidate-cap-hit "$MAX_CANDIDATE_CAP_HIT" \
  --max-malformed-regression "$MAX_MALFORMED_REGRESSION" \
  --max-cap-hit-regression "$MAX_CAP_HIT_REGRESSION"

if [[ "$RUN_AGGREGATE" == "1" ]]; then
  "$PYTHON" -m randopt_lora_lab.prompt_robustness \
    --summary "$OUT_ROOT/cap_stability/summary.json" \
    --out "$OUT_ROOT/aggregate" \
    --split holdout \
    --target-kind aggregate \
    --min-valid-prompts "$MIN_VALID_PROMPTS" \
    --min-lift "$MIN_LIFT" \
    --max-candidate-malformed "$MAX_CANDIDATE_MALFORMED" \
    --max-candidate-cap-hit "$MAX_CANDIDATE_CAP_HIT" \
    --max-malformed-regression "$MAX_MALFORMED_REGRESSION" \
    --max-cap-hit-regression "$MAX_CAP_HIT_REGRESSION"
fi
