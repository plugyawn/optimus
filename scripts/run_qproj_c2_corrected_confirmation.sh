#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run the corrected q-proj activation-spectral c2 dense+vLLM+PEFT confirmation.

Default mode is preflight-only and does not load the model:

  scripts/run_qproj_c2_corrected_confirmation.sh

Run the actual confirmation on a GPU only when explicitly requested:

  MODE=confirm scripts/run_qproj_c2_corrected_confirmation.sh

This differs from run_qproj_c2_exact_replay.sh: it creates a fresh vLLM screen
with the corrected base-healthy prompt set instead of replaying the older
default,reordered,xml source panel.
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
  preflight|confirm)
    ;;
  *)
    usage >&2
    echo "MODE must be either 'preflight' or 'confirm', got: $MODE" >&2
    exit 2
    ;;
esac

PYTHON=${PYTHON:-python}
export OUT_ROOT=${OUT_ROOT:-results/qproj_c2_vllm_shortlist_p64_default_reordered}
export FAMILY=${FAMILY:-activation_spectral_lora_c2}
export TARGETS=${TARGETS:-q_proj}
export POPULATION=${POPULATION:-64}
export PROMPTS=${PROMPTS:-64}
export HOLDOUT_PROMPTS=${HOLDOUT_PROMPTS:-128}
export VLLM_HOLDOUT_PROMPTS=${VLLM_HOLDOUT_PROMPTS:-8}
export SHORTLIST_K=${SHORTLIST_K:-4}
export SHORTLIST_POLICY=${SHORTLIST_POLICY:-default_exact}
export CONFIRM_KS=${CONFIRM_KS:-1,2,4}
export CONFIRM_MAX_K=${CONFIRM_MAX_K:-4}
export CONFIRM_MAX_DENSE_REGRET=${CONFIRM_MAX_DENSE_REGRET:-0.015625}
export VLLM_PROMPT_VARIANTS=${VLLM_PROMPT_VARIANTS:-default,reordered}
export VLLM_REQUIRE_ALL_PROMPT_VARIANTS_VALID=${VLLM_REQUIRE_ALL_PROMPT_VARIANTS_VALID:-1}
export RUN_SCORE_SANITY=${RUN_SCORE_SANITY:-1}
export RUN_SEARCH_QUALITY=${RUN_SEARCH_QUALITY:-1}
RUN_GOAL_AUDIT=${RUN_GOAL_AUDIT:-1}
GOAL_AUDIT_OUT=${GOAL_AUDIT_OUT:-$OUT_ROOT/current_goal_audit}
PREFLIGHT_SUMMARY_OUT=${PREFLIGHT_SUMMARY_OUT:-$OUT_ROOT/preflight_summary.json}
REPLAY_MANIFEST_OUT=${REPLAY_MANIFEST_OUT:-$OUT_ROOT/replay_manifest}

write_preflight_summary() {
  mkdir -p "$OUT_ROOT"
  QPROJ_CONFIRMATION_MODE="$MODE" \
  PREFLIGHT_SUMMARY_OUT="$PREFLIGHT_SUMMARY_OUT" \
  "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

keys = [
    "OUT_ROOT",
    "FAMILY",
    "TARGETS",
    "POPULATION",
    "PROMPTS",
    "HOLDOUT_PROMPTS",
    "VLLM_HOLDOUT_PROMPTS",
    "SHORTLIST_K",
    "SHORTLIST_POLICY",
    "CONFIRM_KS",
    "CONFIRM_MAX_K",
    "CONFIRM_MAX_DENSE_REGRET",
    "VLLM_PROMPT_VARIANTS",
    "VLLM_REQUIRE_ALL_PROMPT_VARIANTS_VALID",
    "RUN_SCORE_SANITY",
    "RUN_SEARCH_QUALITY",
]
summary = {
    "kind": "qproj_c2_corrected_confirmation_preflight",
    "pass": True,
    "mode": os.environ.get("QPROJ_CONFIRMATION_MODE"),
    "config": {key: os.environ.get(key) for key in keys},
}
Path(os.environ["PREFLIGHT_SUMMARY_OUT"]).write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY
}

write_preflight_summary

if [[ "$MODE" == "preflight" ]]; then
  echo "preflight complete: $PREFLIGHT_SUMMARY_OUT"
  echo "preflight complete; run with MODE=confirm to launch the GPU confirmation"
  exit 0
fi

scripts/run_vllm_shortlist_confirmation.sh

goal_audit_status=0
if [[ "$RUN_GOAL_AUDIT" == "1" ]]; then
  if ! QPROJ_REPLAY_ROOT="$OUT_ROOT" OUT="$GOAL_AUDIT_OUT" scripts/run_current_goal_audit.sh; then
    goal_audit_status=1
    echo "current goal audit failed; continuing to replay manifest" >&2
  fi
fi

"$PYTHON" -m randopt_lora_lab.replay_manifest \
  --root "$OUT_ROOT" \
  --out "$REPLAY_MANIFEST_OUT" \
  --mode confirm

exit "$goal_audit_status"
