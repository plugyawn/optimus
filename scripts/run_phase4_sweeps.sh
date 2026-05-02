#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=${HF_HOME:-/root/hf_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/root/hf_cache}
export PYTHONUNBUFFERED=1
export VLLM_USE_DEEP_GEMM=${VLLM_USE_DEEP_GEMM:-0}
export VLLM_DEEP_GEMM_WARMUP=${VLLM_DEEP_GEMM_WARMUP:-skip}

if [ -d .venv ]; then
  # shellcheck disable=SC1091
  . .venv/bin/activate
fi

DEADLINE_UTC=${DEADLINE_UTC:-2026-05-02T20:59:11Z}
PRIOR_RESULTS=${PRIOR_RESULTS:-results/search_anzo_p64,results/halving_anzo_p128,results/phase3_hybrid_covlite_p96,results/phase3_elite_basis_p96,results/sigma_iso_p64_s0p005,results/sigma_iso_p64_s0p02}

seconds_left() {
  local now deadline
  now=$(date -u +%s)
  deadline=$(date -u -d "$DEADLINE_UTC" +%s)
  echo $((deadline - now))
}

enough_time() {
  local min_left="$1"
  [ "$(seconds_left)" -gt "$min_left" ]
}

git_pull() {
  if [ -f /root/.gh_token_randopt_lora_lab ]; then
    local token basic
    token=$(cat /root/.gh_token_randopt_lora_lab)
    basic=$(printf "x-access-token:%s" "$token" | base64 -w0)
    git -c http.extraHeader="AUTHORIZATION: basic $basic" pull --ff-only || true
  else
    git pull --ff-only || true
  fi
}

push_results() {
  local msg="$1"
  git add README.md randopt_lora_lab scripts pyproject.toml || true
  git add -f results || true
  if git diff --cached --quiet; then
    return 0
  fi
  git commit -m "$msg"
  if [ -f /root/.gh_token_randopt_lora_lab ]; then
    local token basic
    token=$(cat /root/.gh_token_randopt_lora_lab)
    basic=$(printf "x-access-token:%s" "$token" | base64 -w0)
    git -c http.extraHeader="AUTHORIZATION: basic $basic" push origin main || true
  else
    git push origin main || true
  fi
}

run_vllm_lora_count() {
  local adapters="$1"
  local prompts="$2"
  local out="results/vllm_lora_bench_a${adapters}_p${prompts}"
  (
    if [ ! -d /root/vllm_venv ]; then
      python3 -m venv /root/vllm_venv
    fi
    # shellcheck disable=SC1091
    . /root/vllm_venv/bin/activate
    pip install -e . --no-deps
    python -m randopt_lora_lab.vllm_lora_bench \
      --out "$out" \
      --model Qwen/Qwen2.5-3B-Instruct \
      --adapters "$adapters" \
      --prompts "$prompts" \
      --rank 8 \
      --sigma 0.02 \
      --targets q_proj,v_proj \
      --max-loras "$adapters" \
      --preload
  ) 2>&1 | tee "results/labrun_vllm_lora_a${adapters}_p${prompts}.log"
  push_results "Add vLLM LoRA ${adapters}-adapter bench"
}

git_pull

for sigma in 0.005 0.01 0.02 0.04; do
  safe_sigma=${sigma//./p}
  if enough_time 900; then
    python -m randopt_lora_lab.experiments search \
      --out "results/anzo_sigma_p64_s${safe_sigma}" \
      --family anzo \
      --population 64 \
      --sigma "$sigma" \
      --prompts 32 \
      --holdout-prompts 32 \
      --batch-size 16 \
      --antithetic
    push_results "Add ANZO sigma ${sigma} search"
  fi
done

if enough_time 1200; then
  python -m randopt_lora_lab.adaptive search \
    --out results/phase4_covlite_sigma001_p128 \
    --mode covlite \
    --basis-source elite,current \
    --prior-results "$PRIOR_RESULTS" \
    --population 128 \
    --rounds 2 \
    --basis-elites 24 \
    --basis-rank 24 \
    --sigma 0.01 \
    --prompts 32 \
    --holdout-prompts 32 \
    --batch-size 16 \
    --antithetic
  push_results "Add phase4 covlite sigma 0.01 adaptive search"
fi

if enough_time 900; then
  run_vllm_lora_count 4 64
fi

if enough_time 900; then
  run_vllm_lora_count 16 32
fi

if enough_time 300; then
  python -m randopt_lora_lab.report --root results --out results/report
  push_results "Add phase4 sweep report"
fi
