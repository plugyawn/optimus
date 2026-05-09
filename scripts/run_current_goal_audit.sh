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

"$PYTHON" -m randopt_lora_lab.goal_audit \
  --reproduction-audit results/paper_style_p128_qwen3b/dense/reproduction_audit/summary.json \
  --parity-report results/spectral_vllm_confirmation_rank32_c1p5_p16_default/parity/summary.json \
  --parity-arm lora \
  --backend-gate results/backend_parity_gate_p64_tokenized_vllm/gate/summary.json \
  --confirmation-gate results/spectral_vllm_confirmation_rank32_c1p5_p16_default/confirmation/summary.json \
  --dense-confirmation-gate results/qproj_c2_vllm_shortlist_p64/shortlist_dense_confirmation/summary.json \
  --search-quality-confirmation results/qproj_c2_vllm_shortlist_p64/search_quality_confirmation/summary.json \
  --family-state-provenance results/family_state_provenance_audit_current/summary.json \
  --multirun-gate results/spectral_vllm_multirun_gate_p16_default/summary.json \
  --prompt-robustness results/prompt_robustness_rank32_top4/summary.json \
  --drift-report results/drift_parity_dense_vs_lora_rank8_p32_sigma001/summary.json \
  --eval-validity results/qproj_c2_vllm_shortlist_p64/confirmed/validity/summary.json \
  --adapter-run results/qproj_c2_vllm_shortlist_p64/vllm \
  --out "$OUT"

echo "goal audit written to $OUT"
