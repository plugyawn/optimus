#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=${HF_HOME:-/root/hf_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/root/hf_cache}
export PYTHONUNBUFFERED=1

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

git_pull

if enough_time 3600; then
  python -m randopt_lora_lab.experiments search \
    --out results/search_iso_s0p01_p512_seed5678 \
    --family isotropic \
    --population 512 \
    --sigma 0.01 \
    --seed 5678 \
    --prompts 32 \
    --holdout-prompts 32 \
    --batch-size 16 \
    --antithetic
  push_results "Add isotropic sigma 0.01 P512 search"
fi

if enough_time 3600; then
  python -m randopt_lora_lab.experiments search \
    --out results/search_anzo_s0p02_p512_seed5678 \
    --family anzo \
    --population 512 \
    --sigma 0.02 \
    --seed 5678 \
    --prompts 32 \
    --holdout-prompts 32 \
    --batch-size 16 \
    --antithetic
  push_results "Add ANZO sigma 0.02 P512 search"
fi

if enough_time 2400; then
  python -m randopt_lora_lab.experiments halving \
    --out results/halving_iso_s0p01_p1024_seed9012 \
    --family isotropic \
    --population 1024 \
    --sigma 0.01 \
    --seed 9012 \
    --prompts 32 \
    --stage-prompts 8 \
    --survivors 64 \
    --holdout-prompts 32 \
    --batch-size 16 \
    --antithetic
  push_results "Add isotropic sigma 0.01 P1024 halving"
fi

if enough_time 2400; then
  python -m randopt_lora_lab.experiments halving \
    --out results/halving_anzo_s0p02_p1024_seed9012 \
    --family anzo \
    --population 1024 \
    --sigma 0.02 \
    --seed 9012 \
    --prompts 32 \
    --stage-prompts 8 \
    --survivors 64 \
    --holdout-prompts 32 \
    --batch-size 16 \
    --antithetic
  push_results "Add ANZO sigma 0.02 P1024 halving"
fi

python -m randopt_lora_lab.report --root results --out results/report
push_results "Add phase5 long-sweep report"
