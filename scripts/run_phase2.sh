#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=${HF_HOME:-/root/hf_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/root/hf_cache}
export PYTHONUNBUFFERED=1

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

git_pull

python -m randopt_lora_lab.experiments sysbench --out results/sysbench_tf3b --prompts 64 --batch-size 16 --batch-sizes 4,8,16,32 --prompt-counts 8,16,32,64 --repeats 2
push_results "Add transformers systems bench"

for sigma in 0.005 0.01 0.02 0.04; do
  safe_sigma=${sigma//./p}
  python -m randopt_lora_lab.experiments search --out "results/sigma_iso_p64_s${safe_sigma}" --family isotropic --population 64 --sigma "$sigma" --prompts 32 --holdout-prompts 32 --batch-size 16 --antithetic
  push_results "Add isotropic sigma ${sigma} search"
done

python -m randopt_lora_lab.experiments halving --out results/halving_iso_p128 --family isotropic --population 128 --sigma 0.02 --prompts 32 --stage-prompts 8 --survivors 16 --holdout-prompts 32 --batch-size 16 --antithetic
push_results "Add isotropic halving run"

python -m randopt_lora_lab.experiments halving --out results/halving_anzo_p128 --family anzo --population 128 --sigma 0.02 --prompts 32 --stage-prompts 8 --survivors 16 --holdout-prompts 32 --batch-size 16 --antithetic
push_results "Add ANZO halving run"

python -m randopt_lora_lab.report --root results --out results/report
push_results "Add phase2 report"
