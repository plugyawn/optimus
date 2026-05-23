#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct}
DATA=${DATA:-data/countdown_generated_1200_seed20260507.json}
OUT_ROOT=${OUT_ROOT:-results/phase8_extra_systems}
FULL_ROOT=${FULL_ROOT:-$OUT_ROOT}
POPULATION=${POPULATION:-512}
PROMPTS=${PROMPTS:-64}
SEED=${SEED:-2468}
RANK=${RANK:-8}
SIGMA=${SIGMA:-0.0075}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-32}

export VLLM_USAGE_STATS_ENABLED=${VLLM_USAGE_STATS_ENABLED:-0}
export XDG_CONFIG_HOME=${XDG_CONFIG_HOME:-/tmp/randopt-xdg-config}
mkdir -p "$OUT_ROOT" "$XDG_CONFIG_HOME"

if [[ ! -f "$DATA" ]]; then
  python -m randopt_lora_lab.make_countdown_data --out "$DATA" --count 1200 --seed 20260507
fi

run_search() {
  local name=$1
  local chunk=$2
  local max_new_tokens=$3
  shift 3
  if [[ -f "$OUT_ROOT/$name/summary.json" ]]; then
    echo "skip $name"
    return
  fi
  python -m randopt_lora_lab.vllm_lora_search \
    --out "$OUT_ROOT/$name" \
    --model "$MODEL" \
    --data "$DATA" \
    --prompts "$PROMPTS" \
    --holdout-prompts 8 \
    --population "$POPULATION" \
    --promote 0 \
    --rank "$RANK" \
    --sigma "$SIGMA" \
    --seed "$SEED" \
    --targets q_proj,v_proj \
    --max-loras "$chunk" \
    --chunk-adapters "$chunk" \
    --max-cpu-loras 4096 \
    --max-new-tokens "$max_new_tokens" \
    --stop-at-answer \
    --antithetic \
    "$@"
}

run_halving() {
  local name=$1
  local stage_prompts=$2
  local survivors=$3
  local chunk=$4
  if [[ -f "$OUT_ROOT/$name/summary.json" ]]; then
    echo "skip $name"
    return
  fi
  python -m randopt_lora_lab.vllm_lora_halving \
    --out "$OUT_ROOT/$name" \
    --model "$MODEL" \
    --data "$DATA" \
    --prompts "$PROMPTS" \
    --stage-prompts "$stage_prompts" \
    --holdout-prompts 8 \
    --population "$POPULATION" \
    --survivors "$survivors" \
    --promote 0 \
    --rank "$RANK" \
    --sigma "$SIGMA" \
    --seed "$SEED" \
    --targets q_proj,v_proj \
    --max-loras "$chunk" \
    --chunk-adapters "$chunk" \
    --max-cpu-loras 4096 \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --stop-at-answer \
    --antithetic
}

run_compare() {
  local full=$1
  local halving=$2
  local name=$3
  if [[ -f "$OUT_ROOT/$name/summary.json" ]]; then
    echo "skip $name"
    return
  fi
  if [[ ! -f "$FULL_ROOT/$full/candidate_summary.jsonl" ]]; then
    echo "skip $name; missing full $FULL_ROOT/$full"
    return
  fi
  python -m randopt_lora_lab.compare_halving \
    --full "$FULL_ROOT/$full" \
    --halving "$OUT_ROOT/$halving" \
    --out "$OUT_ROOT/$name" || true
}

for chunk in ${EXTRA_CHUNKS:-4 12}; do
  run_search "search_chunk${chunk}_p${POPULATION}" "$chunk" "$MAX_NEW_TOKENS"
done

run_search "search_chunk8_p${POPULATION}_tok16" 8 16
run_search "search_chunk8_p${POPULATION}_mbt16384" 8 "$MAX_NEW_TOKENS" --max-num-batched-tokens 16384

run_halving "halving_stage4_surv32_p${POPULATION}" 4 32 8
run_halving "halving_stage8_surv64_p${POPULATION}" 8 64 8
run_compare "search_chunk8_p${POPULATION}" "halving_stage4_surv32_p${POPULATION}" "halving_stage4_surv32_vs_full_chunk8"
run_compare "search_chunk8_p${POPULATION}" "halving_stage8_surv64_p${POPULATION}" "halving_stage8_surv64_vs_full_chunk8"

if [[ "${RUN_P1024:-1}" == "1" ]]; then
  POPULATION=1024 run_search "search_chunk8_p1024" 8 "$MAX_NEW_TOKENS"
fi

python -m randopt_lora_lab.report --root results --out results/report
