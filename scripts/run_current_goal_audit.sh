#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run the current non-GPU completion audit for the superfast LoRA perturbation
search goal.

This does not launch inference or require a GPU. It only reads existing
artifacts and writes a machine-readable audit under OUT.

Default:
  scripts/run_current_goal_audit.sh

Useful override:
  OUT=/tmp/randopt_current_goal_audit scripts/run_current_goal_audit.sh

After corrected q-proj c2 confirmation:
  QPROJ_REPLAY_ROOT=results/qproj_c2_vllm_shortlist_p64_default_reordered \
    scripts/run_current_goal_audit.sh
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

PYTHON=${PYTHON:-python}
OUT=${OUT:-results/current_goal_audit_current}

REPRODUCTION_AUDIT=${REPRODUCTION_AUDIT:-results/paper_style_p128_qwen3b/dense/reproduction_audit/summary.json}
PARITY_REPORT=${PARITY_REPORT:-results/spectral_vllm_confirmation_rank32_c1p5_p16_default/parity/summary.json}
BACKEND_GATE=${BACKEND_GATE:-results/backend_parity_gate_p64_tokenized_vllm/gate/summary.json}
CONFIRMATION_GATE=${CONFIRMATION_GATE:-results/spectral_vllm_confirmation_rank32_c1p5_p16_default/confirmation/summary.json}
QPROJ_REPLAY_ROOT=${QPROJ_REPLAY_ROOT:-results/qproj_c2_vllm_shortlist_p64_default_reordered}
DENSE_CONFIRMATION_GATE=${DENSE_CONFIRMATION_GATE:-$QPROJ_REPLAY_ROOT/shortlist_dense_confirmation/summary.json}
SEARCH_QUALITY_CONFIRMATION=${SEARCH_QUALITY_CONFIRMATION:-$QPROJ_REPLAY_ROOT/search_quality_confirmation/summary.json}
FAMILY_STATE_PROVENANCE=${FAMILY_STATE_PROVENANCE:-$QPROJ_REPLAY_ROOT/family_state_provenance_audit/summary.json}
MULTIRUN_GATE=${MULTIRUN_GATE:-results/spectral_vllm_multirun_gate_p16_default/summary.json}
DENSE_REFERENCED_MULTIRUN_GATE=${DENSE_REFERENCED_MULTIRUN_GATE:-results/qproj_c2_dense_referenced_multirun_gate/summary.json}
PROMPT_ROBUSTNESS=${PROMPT_ROBUSTNESS:-results/prompt_robustness_rank32_top4/summary.json}
DRIFT_REPORT=${DRIFT_REPORT:-results/drift_parity_dense_vs_lora_rank8_p32_sigma001/summary.json}
EVAL_VALIDITY=${EVAL_VALIDITY:-$QPROJ_REPLAY_ROOT/confirmed/validity/summary.json}
if [[ -z "${SCORE_SANITY:-}" ]]; then
  if [[ -f "$QPROJ_REPLAY_ROOT/score_sanity/summary.json" ]]; then
    SCORE_SANITY=$QPROJ_REPLAY_ROOT/score_sanity/summary.json
  elif [[ -f "$QPROJ_REPLAY_ROOT/score_sanity_current/summary.json" ]]; then
    SCORE_SANITY=$QPROJ_REPLAY_ROOT/score_sanity_current/summary.json
  else
    SCORE_SANITY=$QPROJ_REPLAY_ROOT/score_sanity/summary.json
  fi
fi
ADAPTER_RUN=${ADAPTER_RUN:-$QPROJ_REPLAY_ROOT/vllm}

"$PYTHON" -m randopt_lora_lab.goal_audit \
  --reproduction-audit "$REPRODUCTION_AUDIT" \
  --parity-report "$PARITY_REPORT" \
  --parity-arm lora \
  --backend-gate "$BACKEND_GATE" \
  --confirmation-gate "$CONFIRMATION_GATE" \
  --dense-confirmation-gate "$DENSE_CONFIRMATION_GATE" \
  --search-quality-confirmation "$SEARCH_QUALITY_CONFIRMATION" \
  --family-state-provenance "$FAMILY_STATE_PROVENANCE" \
  --multirun-gate "$MULTIRUN_GATE" \
  --dense-referenced-multirun-gate "$DENSE_REFERENCED_MULTIRUN_GATE" \
  --prompt-robustness "$PROMPT_ROBUSTNESS" \
  --drift-report "$DRIFT_REPORT" \
  --eval-validity "$EVAL_VALIDITY" \
  --score-sanity "$SCORE_SANITY" \
  --adapter-run "$ADAPTER_RUN" \
  --out "$OUT"

echo "goal audit written to $OUT"
