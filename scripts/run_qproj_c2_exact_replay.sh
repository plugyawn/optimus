#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Replay the current q-proj activation-spectral c2 shortlist with exact PEFT
confirmation and saved vLLM family-state provenance.

Default mode is preflight-only and does not load the model:

  scripts/run_qproj_c2_exact_replay.sh

Run the actual confirmation on a GPU only when explicitly requested:

  MODE=confirm scripts/run_qproj_c2_exact_replay.sh

Confirmation mode automatically runs the current non-GPU goal audit against
OUT_ROOT when the replay finishes. Disable with RUN_GOAL_AUDIT=0.

Useful overrides:

  SOURCE_ROOT=results/qproj_c2_vllm_shortlist_p64
  OUT_ROOT=results/qproj_c2_vllm_shortlist_p64_default_exact_k4
  SHORTLIST_K=4
  CONFIRM_MAX_DENSE_REGRET=0.015625
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

MODE=${MODE:-preflight}
case "$MODE" in
  preflight)
    export PREFLIGHT_ONLY=1
    ;;
  confirm)
    export PREFLIGHT_ONLY=0
    ;;
  *)
    usage >&2
    echo "MODE must be either 'preflight' or 'confirm', got: $MODE" >&2
    exit 2
    ;;
esac

export SOURCE_ROOT=${SOURCE_ROOT:-results/qproj_c2_vllm_shortlist_p64}
export OUT_ROOT=${OUT_ROOT:-results/qproj_c2_vllm_shortlist_p64_default_exact_k4}
export FAMILY=${FAMILY:-activation_spectral_lora_c2}
export TARGETS=${TARGETS:-q_proj}
export SEED=${SEED:-20260507}
export DATA=${DATA:-data/countdown_generated_1200_seed20260507.json}
export SHORTLIST_POLICY=${SHORTLIST_POLICY:-default_exact}
export SHORTLIST_K=${SHORTLIST_K:-4}
export CONFIRM_KS=${CONFIRM_KS:-1,2,4}
export CONFIRM_MAX_K=${CONFIRM_MAX_K:-$SHORTLIST_K}
export CONFIRM_MAX_DENSE_REGRET=${CONFIRM_MAX_DENSE_REGRET:-0.015625}
export CONFIRM_MIN_FULL_SPEEDUP=${CONFIRM_MIN_FULL_SPEEDUP:-1.0}
RUN_GOAL_AUDIT=${RUN_GOAL_AUDIT:-1}
GOAL_AUDIT_OUT=${GOAL_AUDIT_OUT:-$OUT_ROOT/current_goal_audit}

scripts/run_existing_vllm_shortlist_confirmation.sh

if [[ "$MODE" == "confirm" && "$RUN_GOAL_AUDIT" == "1" ]]; then
  QPROJ_REPLAY_ROOT="$OUT_ROOT" OUT="$GOAL_AUDIT_OUT" scripts/run_current_goal_audit.sh
fi
