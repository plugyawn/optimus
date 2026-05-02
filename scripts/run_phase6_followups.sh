#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=${HF_HOME:-/root/hf_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/root/hf_cache}
export PYTHONUNBUFFERED=1
export VLLM_USE_DEEP_GEMM=${VLLM_USE_DEEP_GEMM:-0}

if [ -d .venv ]; then
  # shellcheck disable=SC1091
  . .venv/bin/activate
fi

DEADLINE_UTC=${DEADLINE_UTC:-2026-05-02T20:59:11Z}

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
    local token
    token=$(cat /root/.gh_token_randopt_lora_lab)
    git fetch "https://x-access-token:${token}@github.com/plugyawn/randopt-lora-lab.git" main:refs/remotes/origin/main || true
    git rebase origin/main || true
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
    local token
    token=$(cat /root/.gh_token_randopt_lora_lab)
    git fetch "https://x-access-token:${token}@github.com/plugyawn/randopt-lora-lab.git" main:refs/remotes/origin/main || true
    git rebase origin/main || true
    git push "https://x-access-token:${token}@github.com/plugyawn/randopt-lora-lab.git" HEAD:main || true
  else
    git push origin main || true
  fi
}

run_optional() {
  local msg="$1"
  shift
  set +e
  "$@"
  local rc=$?
  set -e
  if [ "$rc" -ne 0 ]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] optional command failed rc=$rc: $msg"
  fi
  push_results "$msg"
}

git_pull

if enough_time 900; then
  run_optional "Add stop-at-answer mixed LoRA bench" python -m randopt_lora_lab.vllm_lora_bench \
    --out results/vllm_lora_mixed_stop_a16_p32 \
    --model Qwen/Qwen2.5-3B-Instruct \
    --adapters 16 \
    --prompts 32 \
    --rank 8 \
    --sigma 0.02 \
    --targets q_proj,v_proj \
    --max-loras 16 \
    --preload \
    --mixed-batch \
    --stop-at-answer
fi

if enough_time 900; then
  run_optional "Add 32-adapter mixed LoRA bench" python -m randopt_lora_lab.vllm_lora_bench \
    --out results/vllm_lora_mixed_stop_a32_p32 \
    --model Qwen/Qwen2.5-3B-Instruct \
    --adapters 32 \
    --prompts 32 \
    --rank 8 \
    --sigma 0.02 \
    --targets q_proj,v_proj \
    --max-loras 32 \
    --preload \
    --mixed-batch \
    --stop-at-answer
fi

if enough_time 900; then
  run_optional "Add 64-prompt mixed LoRA bench" python -m randopt_lora_lab.vllm_lora_bench \
    --out results/vllm_lora_mixed_stop_a16_p64 \
    --model Qwen/Qwen2.5-3B-Instruct \
    --adapters 16 \
    --prompts 64 \
    --rank 8 \
    --sigma 0.02 \
    --targets q_proj,v_proj \
    --max-loras 16 \
    --preload \
    --mixed-batch \
    --stop-at-answer
fi

if enough_time 1200; then
  run_optional "Add vLLM mixed LoRA P512 search" python -m randopt_lora_lab.vllm_lora_search \
    --out results/vllm_lora_search_iso_s0p01_p512_stop \
    --model Qwen/Qwen2.5-3B-Instruct \
    --family isotropic \
    --population 512 \
    --sigma 0.01 \
    --seed 5678 \
    --prompts 32 \
    --holdout-prompts 64 \
    --promote 8 \
    --rank 8 \
    --targets q_proj,v_proj \
    --max-loras 32 \
    --chunk-adapters 32 \
    --max-cpu-loras 1024 \
    --antithetic \
    --stop-at-answer
fi

if enough_time 1200; then
  run_optional "Add vLLM mixed LoRA sigma 0.0075 search" python -m randopt_lora_lab.vllm_lora_search \
    --out results/vllm_lora_search_iso_s0p0075_p512_stop \
    --model Qwen/Qwen2.5-3B-Instruct \
    --family isotropic \
    --population 512 \
    --sigma 0.0075 \
    --seed 2468 \
    --prompts 32 \
    --holdout-prompts 64 \
    --promote 8 \
    --rank 8 \
    --targets q_proj,v_proj \
    --max-loras 32 \
    --chunk-adapters 32 \
    --max-cpu-loras 1024 \
    --antithetic \
    --stop-at-answer
fi

if enough_time 2700; then
  python -m randopt_lora_lab.experiments search \
    --out results/search_iso_s0p0075_p512_seed2468 \
    --family isotropic \
    --population 512 \
    --sigma 0.0075 \
    --seed 2468 \
    --prompts 32 \
    --holdout-prompts 64 \
    --batch-size 16 \
    --antithetic
  push_results "Add isotropic sigma 0.0075 P512 search"
fi

if enough_time 2700; then
  python -m randopt_lora_lab.experiments search \
    --out results/search_iso_s0p0125_p512_seed2468 \
    --family isotropic \
    --population 512 \
    --sigma 0.0125 \
    --seed 2468 \
    --prompts 32 \
    --holdout-prompts 64 \
    --batch-size 16 \
    --antithetic
  push_results "Add isotropic sigma 0.0125 P512 search"
fi

if enough_time 2700; then
  python -m randopt_lora_lab.experiments search \
    --out results/search_anzo_s0p015_p512_seed2468 \
    --family anzo \
    --population 512 \
    --sigma 0.015 \
    --seed 2468 \
    --prompts 32 \
    --holdout-prompts 64 \
    --batch-size 16 \
    --antithetic
  push_results "Add ANZO sigma 0.015 P512 search"
fi

if enough_time 2700; then
  python -m randopt_lora_lab.experiments search \
    --out results/search_anzo_s0p025_p512_seed2468 \
    --family anzo \
    --population 512 \
    --sigma 0.025 \
    --seed 2468 \
    --prompts 32 \
    --holdout-prompts 64 \
    --batch-size 16 \
    --antithetic
  push_results "Add ANZO sigma 0.025 P512 search"
fi

if enough_time 2400; then
  python -m randopt_lora_lab.adaptive search \
    --out results/phase6_winner_covlite_p256 \
    --mode covlite \
    --basis-source elite,current \
    --prior-results 'results/halving_iso_s0p01_p256,results/sigma_iso_p64_s0p01,results/search_iso_s0p01_p512_seed5678,results/halving_anzo_p128,results/search_anzo_s0p02_p512_seed5678,results/anzo_sigma_p64_s0p02' \
    --population 128 \
    --rounds 2 \
    --basis-elites 24 \
    --basis-rank 24 \
    --sigma 0.005 \
    --prompts 32 \
    --holdout-prompts 64 \
    --batch-size 16 \
    --antithetic
  push_results "Add phase6 winner-seeded covlite search"
fi

python -m randopt_lora_lab.report --root results --out results/report
push_results "Add phase6 follow-up report"
