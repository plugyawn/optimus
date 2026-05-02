#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=${HF_HOME:-/root/hf_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/root/hf_cache}
export PYTHONUNBUFFERED=1

if [ -d .venv ]; then
  # shellcheck disable=SC1091
  . .venv/bin/activate
fi

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

PRIOR_RESULTS=${PRIOR_RESULTS:-results/search_anzo_p64,results/halving_anzo_p128,results/sigma_iso_p64_s0p02,results/search_iso_p64}

python -m randopt_lora_lab.adaptive search \
  --out results/phase3_hybrid_covlite_p96 \
  --mode hybrid-covlite \
  --basis-source hybrid,current \
  --prior-results "$PRIOR_RESULTS" \
  --population 96 \
  --rounds 2 \
  --basis-elites 16 \
  --basis-rank 16 \
  --sigma 0.02 \
  --prompts 32 \
  --holdout-prompts 32 \
  --batch-size 16 \
  --antithetic
push_results "Add phase3 hybrid covlite adaptive search"

python -m randopt_lora_lab.adaptive search \
  --out results/phase3_elite_basis_p96 \
  --mode elite-basis \
  --basis-source hybrid,current \
  --prior-results "$PRIOR_RESULTS" \
  --population 96 \
  --rounds 2 \
  --basis-elites 16 \
  --basis-rank 16 \
  --sigma 0.02 \
  --prompts 32 \
  --holdout-prompts 32 \
  --batch-size 16 \
  --antithetic
push_results "Add phase3 elite basis adaptive search"

python -m randopt_lora_lab.report --root results --out results/report
push_results "Add phase3 adaptive report"
