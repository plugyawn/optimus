#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=${HF_HOME:-/root/hf_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/root/hf_cache}
export PYTHONUNBUFFERED=1

push_results() {
  local msg="$1"
  git add results README.md randopt_lora_lab scripts pyproject.toml || true
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

git pull --ff-only || true

python -m randopt_lora_lab.experiments oracle --out results/oracle --prompts 16 --batch-size 16
push_results "Add oracle run"

python -m randopt_lora_lab.experiments search --out results/search_iso_p32 --family isotropic --population 32 --prompts 32 --holdout-prompts 32 --batch-size 16 --antithetic
push_results "Add isotropic p32 run"

python -m randopt_lora_lab.experiments search --out results/search_iso_p64 --family isotropic --population 64 --prompts 32 --holdout-prompts 32 --batch-size 16 --antithetic
push_results "Add isotropic p64 run"

python -m randopt_lora_lab.experiments search --out results/search_anzo_p32 --family anzo --population 32 --prompts 32 --holdout-prompts 32 --batch-size 16 --antithetic
push_results "Add ANZO p32 run"

python -m randopt_lora_lab.experiments search --out results/search_anzo_p64 --family anzo --population 64 --prompts 32 --holdout-prompts 32 --batch-size 16 --antithetic
push_results "Add ANZO p64 run"

python -m randopt_lora_lab.report --root results --out results/report
push_results "Add report"
