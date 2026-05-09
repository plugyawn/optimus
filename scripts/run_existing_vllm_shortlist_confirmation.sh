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

Preflight without loading the model:
  PREFLIGHT_ONLY=1 \
  SOURCE_ROOT=results/qproj_c2_vllm_shortlist_p64 \
  OUT_ROOT=/tmp/qproj_c2_default_exact_k4_preflight \
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
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}

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

"$PYTHON" - "$SOURCE_ROOT" "$OUT_ROOT" "$FAMILY" "$SHORTLIST_K" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
out = Path(sys.argv[2])
family = sys.argv[3]
shortlist_k = int(sys.argv[4])

def read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}

def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

shortlist = read_jsonl(out / f"shortlist_top{shortlist_k}.jsonl")
vllm_rows = read_jsonl(out / "vllm" / "candidate_summary.jsonl")
vllm_candidates = {str(row.get("candidate")) for row in vllm_rows}
missing = [row.get("candidate") for row in shortlist if str(row.get("candidate")) not in vllm_candidates]
wrong_family = [row.get("candidate") for row in shortlist if not str(row.get("candidate", "")).startswith(family + ":")]
source_state = source / "vllm" / "family_state.pt"
copied_state = out / "vllm" / "family_state.pt"
summary = {
    "kind": "existing_vllm_shortlist_confirmation_preflight",
    "source_root": str(source),
    "out_root": str(out),
    "family": family,
    "shortlist_k": shortlist_k,
    "shortlist_rows": len(shortlist),
    "shortlist_policy": shortlist[0].get("selector_union_policy") if shortlist else None,
    "vllm_candidates": len(vllm_rows),
    "missing_shortlist_candidates": missing,
    "wrong_family_candidates": wrong_family,
    "source_vllm_summary": read_json(source / "vllm" / "summary.json"),
    "out_vllm_summary": read_json(out / "vllm" / "summary.json"),
    "dense_summary_present": (out / "dense" / "summary.json").exists(),
    "vllm_family_state_sha256": digest(copied_state) if copied_state.exists() else None,
    "source_family_state_sha256": digest(source_state) if source_state.exists() else None,
}
checks = [
    ("shortlist_count_matches", len(shortlist) == shortlist_k),
    ("shortlist_candidates_in_vllm_panel", not missing),
    ("shortlist_candidates_match_family", not wrong_family),
    ("dense_summary_present", summary["dense_summary_present"]),
    ("source_family_state_present", source_state.exists()),
    ("copied_family_state_present", copied_state.exists()),
    (
        "copied_family_state_matches_source",
        source_state.exists() and copied_state.exists() and summary["source_family_state_sha256"] == summary["vllm_family_state_sha256"],
    ),
]
summary["checks"] = [{"check": name, "passed": bool(passed)} for name, passed in checks]
summary["pass"] = all(passed for _, passed in checks)
summary["failed"] = [name for name, passed in checks if not passed]
(out / "preflight_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
if not summary["pass"]:
    raise SystemExit(1)
PY

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
  echo "preflight complete: $OUT_ROOT/preflight_summary.json"
  exit 0
fi

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
