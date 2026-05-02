#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=${HF_HOME:-/root/hf_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/root/hf_cache}
export PYTHONUNBUFFERED=1

python -m randopt_lora_lab.experiments oracle --out results/oracle --prompts 16 --batch-size 16
python -m randopt_lora_lab.experiments search --out results/search_iso_p32 --family isotropic --population 32 --prompts 32 --holdout-prompts 32 --batch-size 16 --antithetic
python -m randopt_lora_lab.experiments search --out results/search_iso_p64 --family isotropic --population 64 --prompts 32 --holdout-prompts 32 --batch-size 16 --antithetic
python -m randopt_lora_lab.experiments search --out results/search_anzo_p32 --family anzo --population 32 --prompts 32 --holdout-prompts 32 --batch-size 16 --antithetic
python -m randopt_lora_lab.experiments search --out results/search_anzo_p64 --family anzo --population 64 --prompts 32 --holdout-prompts 32 --batch-size 16 --antithetic
python -m randopt_lora_lab.report --root results --out results/report
