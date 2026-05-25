#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate
export PYTHONUNBUFFERED=1
export VLLM_USAGE_STATS_ENABLED=${VLLM_USAGE_STATS_ENABLED:-0}
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-ERROR}
export XDG_CONFIG_HOME=${XDG_CONFIG_HOME:-/tmp/optimus-xdg-config}
mkdir -p "$XDG_CONFIG_HOME" results/prime_smoke

python -m pytest -q \
  tests/test_optimus_package.py \
  tests/test_optimus_hooks.py \
  tests/test_optimus_gpu_suite.py \
  tests/test_optimus_systems_report.py

optimus make-countdown-data \
  --out data/countdown_prime_smoke_64.json \
  --count 64 \
  --seed 20260523

optimus bench \
  --backend vllm \
  --method lora \
  --out results/prime_smoke/prepare_only \
  --data data/countdown_prime_smoke_64.json \
  --adapters 2 \
  --prompts 4 \
  --rank 8 \
  --sigma 0.0075 \
  --targets q_proj,v_proj \
  --adapter-dtype bfloat16 \
  --prepare-only

if [[ "${RUN_VLLM_SMOKE:-1}" == "1" ]]; then
  optimus bench \
    --backend vllm \
    --method lora \
    --out results/prime_smoke/vllm_bench \
    --data data/countdown_prime_smoke_64.json \
    --adapters 2 \
    --prompts 4 \
    --rank 8 \
    --sigma 0.0075 \
    --targets q_proj,v_proj \
    --max-loras 2 \
    --max-cpu-loras 16 \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}" \
    --max-new-tokens 16 \
    --max-model-len 1024 \
    --stop-at-answer \
    --mixed-batch \
    --skip-sequential \
    --no-include-base
fi

cat > results/prime_smoke/report.md <<'EOF'
# Prime Smoke Report

The smoke workflow completed. Inspect `results/prime_smoke/prepare_only` and
`results/prime_smoke/vllm_bench` for adapter materialization and vLLM execution
outputs.
EOF
