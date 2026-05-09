#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Confirm a shortlist from an existing dense+vLLM panel without mutating the
source run. This is the safe replay path for activation-spectral panels because
it reuses SOURCE_ROOT/vllm/family_state.pt during PEFT confirmation.

Required:
  SOURCE_ROOT=results/qproj_c2_vllm_shortlist_p64
  OUT_ROOT=results/qproj_c2_vllm_shortlist_p64_default_exact_k4

Typical:
  SOURCE_ROOT=results/qproj_c2_vllm_shortlist_p64 \
  OUT_ROOT=results/qproj_c2_vllm_shortlist_p64_default_exact_k4 \
  FAMILY=activation_spectral_lora_c2 \
  TARGETS=q_proj \
  SHORTLIST_POLICY=default_exact \
  SHORTLIST_K=4 \
  scripts/run_existing_vllm_shortlist_confirmation.sh
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

SOURCE_ROOT=${SOURCE_ROOT:?SOURCE_ROOT is required}
OUT_ROOT=${OUT_ROOT:?OUT_ROOT is required}
PYTHON=${PYTHON:-python}

MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct}
DATA=${DATA:-data/countdown_generated_1200_seed20260507.json}
FAMILY=${FAMILY:-activation_spectral_lora_c2}
TARGETS=${TARGETS:-q_proj}
SEED=${SEED:-20260507}
PROMPTS=${PROMPTS:-64}
HOLDOUT_PROMPTS=${HOLDOUT_PROMPTS:-128}
SHORTLIST_POLICY=${SHORTLIST_POLICY:-default_exact}
SHORTLIST_K=${SHORTLIST_K:-4}
RANK=${RANK:-32}
SIGMA=${SIGMA:-0.001}
SIGMA_VALUES=${SIGMA_VALUES:-0.0005,0.001,0.002}
ENSEMBLE_KS=${ENSEMBLE_KS:-1,4}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
HF_BATCH_SIZE=${HF_BATCH_SIZE:-16}
CONFIRM_KS=${CONFIRM_KS:-1,2,4}
CONFIRM_MAX_K=${CONFIRM_MAX_K:-$SHORTLIST_K}
CONFIRM_MAX_DENSE_REGRET=${CONFIRM_MAX_DENSE_REGRET:-0.015625}
CONFIRM_MIN_FULL_SPEEDUP=${CONFIRM_MIN_FULL_SPEEDUP:-1.0}

export PYTHONUNBUFFERED=1

if [[ ! -f "$SOURCE_ROOT/dense/candidate_summary.jsonl" ]]; then
  echo "missing source dense candidate summary: $SOURCE_ROOT/dense/candidate_summary.jsonl" >&2
  exit 1
fi
if [[ ! -f "$SOURCE_ROOT/vllm/candidate_summary.jsonl" ]]; then
  echo "missing source vLLM candidate summary: $SOURCE_ROOT/vllm/candidate_summary.jsonl" >&2
  exit 1
fi
if [[ ! -f "$SOURCE_ROOT/vllm/family_state.pt" ]]; then
  echo "missing source vLLM family state: $SOURCE_ROOT/vllm/family_state.pt" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"
rm -rf "$OUT_ROOT/dense" "$OUT_ROOT/vllm"
cp -a "$SOURCE_ROOT/dense" "$OUT_ROOT/dense"
cp -a "$SOURCE_ROOT/vllm" "$OUT_ROOT/vllm"

"$PYTHON" -m randopt_lora_lab.selector_union_audit shortlist \
  --run "$OUT_ROOT" \
  --out "$OUT_ROOT/shortlist_top${SHORTLIST_K}.jsonl" \
  --policy "$SHORTLIST_POLICY" \
  --k "$SHORTLIST_K"

search_args=()
if [[ -n "$SIGMA_VALUES" ]]; then
  search_args+=(--sigma-values "$SIGMA_VALUES")
fi
if [[ -n "$ENSEMBLE_KS" ]]; then
  search_args+=(--ensemble-ks "$ENSEMBLE_KS")
fi

"$PYTHON" -m randopt_lora_lab.experiments search \
  --out "$OUT_ROOT/confirmed" \
  --model "$MODEL" \
  --data "$DATA" \
  --perturbation-backend lora \
  --family "$FAMILY" \
  --candidate-file "$OUT_ROOT/shortlist_top${SHORTLIST_K}.jsonl" \
  --family-state-file "$OUT_ROOT/vllm/family_state.pt" \
  --population "$SHORTLIST_K" \
  --prompts "$PROMPTS" \
  --holdout-prompts "$HOLDOUT_PROMPTS" \
  --promote "$SHORTLIST_K" \
  --rank "$RANK" \
  --sigma "$SIGMA" \
  --seed "$SEED" \
  --targets "$TARGETS" \
  --batch-size "$HF_BATCH_SIZE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --stop-at-answer \
  "${search_args[@]}"

"$PYTHON" -m randopt_lora_lab.result_validity \
  --run "$OUT_ROOT/confirmed" \
  --out "$OUT_ROOT/confirmed/validity"

"$PYTHON" -m randopt_lora_lab.shortlist_dense_confirmation \
  --dense "$OUT_ROOT/dense" \
  --confirmed "$OUT_ROOT/confirmed" \
  --proposal "$OUT_ROOT/vllm" \
  --candidate-file "$OUT_ROOT/shortlist_top${SHORTLIST_K}.jsonl" \
  --out "$OUT_ROOT/shortlist_dense_confirmation" \
  --ks "$CONFIRM_KS" \
  --proposal-score-col selection_score \
  --max-confirm-k "$CONFIRM_MAX_K" \
  --max-dense-regret "$CONFIRM_MAX_DENSE_REGRET" \
  --min-full-without-dense-load-speedup "$CONFIRM_MIN_FULL_SPEEDUP"

"$PYTHON" -m randopt_lora_lab.family_state_provenance_audit \
  --root "$OUT_ROOT" \
  --out "$OUT_ROOT/family_state_provenance_audit"

"$PYTHON" -m randopt_lora_lab.search_quality_confirmation \
  --root "$OUT_ROOT" \
  --out "$OUT_ROOT/search_quality_confirmation" \
  --max-confirm-k "$CONFIRM_MAX_K" \
  --min-full-speedup "$CONFIRM_MIN_FULL_SPEEDUP" \
  --min-holdout-delta 0.0
