#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=${HF_HOME:-/root/hf_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/root/hf_cache}
export PYTHONUNBUFFERED=1

if [ ! -d /root/vllm_venv ]; then
  python3 -m venv /root/vllm_venv
fi

. /root/vllm_venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install vllm
pip install -e . --no-deps

python -m randopt_lora_lab.vllm_probe --out results/vllm_base_probe --prompts 128 --max-new-tokens 32

git add -f results/vllm_base_probe results/labrun_vllm.log || true
git add randopt_lora_lab scripts || true
if ! git diff --cached --quiet; then
  git commit -m "Add vLLM base throughput probe"
  if [ -f /root/.gh_token_randopt_lora_lab ]; then
    token=$(cat /root/.gh_token_randopt_lora_lab)
    basic=$(printf "x-access-token:%s" "$token" | base64 -w0)
    git -c http.extraHeader="AUTHORIZATION: basic $basic" push origin main || true
  else
    git push origin main || true
  fi
fi
