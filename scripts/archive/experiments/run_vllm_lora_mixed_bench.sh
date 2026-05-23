#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=${HF_HOME:-/root/hf_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/root/hf_cache}
export PYTHONUNBUFFERED=1
export VLLM_USE_DEEP_GEMM=${VLLM_USE_DEEP_GEMM:-0}
export VLLM_DEEP_GEMM_WARMUP=${VLLM_DEEP_GEMM_WARMUP:-skip}

if [ ! -d /root/vllm_venv ]; then
  python3 -m venv /root/vllm_venv
fi

# shellcheck disable=SC1091
. /root/vllm_venv/bin/activate
pip install -e . --no-deps

mkdir -p results
python -m randopt_lora_lab.vllm_lora_bench \
  --out results/vllm_lora_mixed_bench \
  --model Qwen/Qwen2.5-3B-Instruct \
  --adapters 16 \
  --prompts 32 \
  --rank 8 \
  --sigma 0.02 \
  --targets q_proj,v_proj \
  --max-loras 16 \
  --preload \
  --mixed-batch \
  "$@" 2>&1 | tee results/labrun_vllm_lora_mixed_bench.log
