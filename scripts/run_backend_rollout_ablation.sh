#!/usr/bin/env bash
set -euo pipefail

DATA=${DATA:-data/countdown_generated_1200_seed20260507.json}
MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct}
ROOT=${ROOT:-results/backend_rollout_ablation_p16}
PROMPTS=${PROMPTS:-8}
SEED=${SEED:-4242}
RANK=${RANK:-8}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-32}
TARGETS=${TARGETS:-q_proj,v_proj}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.72}

CANDIDATE_A=${CANDIDATE_A:-factor_gaussian_lora:seed509771609:s0.0075:sign-1}
CANDIDATE_B=${CANDIDATE_B:-factor_gaussian_lora:seed1019282515:s0.0075:sign1}

export PYTHONUNBUFFERED=1
export VLLM_USAGE_STATS_ENABLED=${VLLM_USAGE_STATS_ENABLED:-0}
export XDG_CONFIG_HOME=${XDG_CONFIG_HOME:-/tmp/randopt-xdg-config}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
mkdir -p "$ROOT" "$XDG_CONFIG_HOME"

run_probe() {
  local name="$1"
  shift
  python -m randopt_lora_lab.backend_rollout_probe \
    --out "$ROOT/$name" \
    --model "$MODEL" \
    --data "$DATA" \
    --prompts "$PROMPTS" \
    --seed "$SEED" \
    --rank "$RANK" \
    --targets "$TARGETS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --include-zero \
    --candidate "$CANDIDATE_A" \
    --candidate "$CANDIDATE_B" \
    "$@"
}

run_probe default --stop-at-answer
run_probe enforce_eager --stop-at-answer --enforce-eager
run_probe adapter_fp32 --stop-at-answer --adapter-dtype float32
run_probe adapter_fp16 --stop-at-answer --adapter-dtype float16
run_probe vllm_fp16 --stop-at-answer --vllm-dtype float16
run_probe no_stop

python -m randopt_lora_lab.backend_rollout_ablation_report \
  --root "$ROOT" \
  --out "$ROOT/report"
