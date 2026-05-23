#!/usr/bin/env bash
set -euo pipefail

UPSTREAM_DIR=${UPSTREAM_DIR:-external/RandOpt}
UPSTREAM_REPO=${UPSTREAM_REPO:-https://github.com/sunrainyg/RandOpt.git}
UPSTREAM_REF=${UPSTREAM_REF:-main}

CUDA_DEVICES=${CUDA_DEVICES:-0}
MODEL=${MODEL:-allenai/Olmo-3-7B-Instruct}
DATASET=${DATASET:-countdown}
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-data/countdown/countdown.json}
TEST_DATA_PATH=${TEST_DATA_PATH:-data/countdown/countdown.json}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-200}
TEST_SAMPLES=${TEST_SAMPLES:-}
POPULATION_SIZE=${POPULATION_SIZE:-5000}
TOP_K_RATIOS=${TOP_K_RATIOS:-0.04,0.01,0.05,0.1}
SIGMA_VALUES=${SIGMA_VALUES:-0.0005,0.001,0.002}
MAX_TOKENS=${MAX_TOKENS:-1024}
PRECISION=${PRECISION:-bfloat16}
TP=${TP:-1}
NUM_ENGINES=${NUM_ENGINES:-}
GLOBAL_SEED=${GLOBAL_SEED:-42}
EXPERIMENT_DIR=${EXPERIMENT_DIR:-randopt-upstream-countdown}

if [[ ! -d "$UPSTREAM_DIR/.git" ]]; then
  mkdir -p "$(dirname "$UPSTREAM_DIR")"
  git clone "$UPSTREAM_REPO" "$UPSTREAM_DIR"
fi

git -C "$UPSTREAM_DIR" fetch --quiet origin "$UPSTREAM_REF"
git -C "$UPSTREAM_DIR" checkout --quiet FETCH_HEAD

if [[ -z "$NUM_ENGINES" ]]; then
  NUM_GPUS="$(awk -F',' '{print NF}' <<< "$CUDA_DEVICES")"
  NUM_ENGINES=$((NUM_GPUS / TP))
fi

args=(
  --dataset "$DATASET"
  --train_data_path "$TRAIN_DATA_PATH"
  --test_data_path "$TEST_DATA_PATH"
  --model_name "$MODEL"
  --num_engines "$NUM_ENGINES"
  --tp "$TP"
  --train_samples "$TRAIN_SAMPLES"
  --precision "$PRECISION"
  --population_size "$POPULATION_SIZE"
  --top_k_ratios "$TOP_K_RATIOS"
  --sigma_values "$SIGMA_VALUES"
  --max_tokens "$MAX_TOKENS"
  --global_seed "$GLOBAL_SEED"
  --experiment_dir "$EXPERIMENT_DIR"
  --cuda_devices "$CUDA_DEVICES"
)

if [[ -n "$TEST_SAMPLES" ]]; then
  args+=(--test_samples "$TEST_SAMPLES")
fi

cd "$UPSTREAM_DIR"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export VLLM_NO_USAGE_STATS=1
python3 randopt.py "${args[@]}"
