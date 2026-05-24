#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SSH_TARGET:-}" ]]; then
  echo "SSH_TARGET is required, for example root@1.2.3.4" >&2
  exit 2
fi

MODE=${MODE:-smoke}
REMOTE_ROOT=${REMOTE_ROOT:-optimus}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
DATA_PARALLEL_SIZE=${DATA_PARALLEL_SIZE:-1}
POPULATIONS=${POPULATIONS:-"1024 4096"}
MODEL=${MODEL:-Qwen/Qwen3-4B-Instruct-2507}
MODEL_TEMPLATE=${MODEL_TEMPLATE:-}
TASKS=${TASKS:-ifeval}
PROMPT_VARIANTS=${PROMPT_VARIANTS:-bare}
PROMPT_INPUT=${PROMPT_INPUT:-text}
USE_CHAT_TEMPLATE=${USE_CHAT_TEMPLATE:-1}
MAX_BASE_MALFORMED_FOR_SELECTION=${MAX_BASE_MALFORMED_FOR_SELECTION:-0.05}
MAX_BASE_CAP_HIT_FOR_SELECTION=${MAX_BASE_CAP_HIT_FOR_SELECTION:-0.05}
MIN_SELECTION_PROMPT_VARIANTS=${MIN_SELECTION_PROMPT_VARIANTS:-1}
REQUIRE_ALL_PROMPT_VARIANTS_VALID=${REQUIRE_ALL_PROMPT_VARIANTS_VALID:-0}
LIGHTEVAL_USE_CHAT_TEMPLATE=${LIGHTEVAL_USE_CHAT_TEMPLATE:-1}
LOCAL_BUNDLE=${LOCAL_BUNDLE:-}
FETCH_RESULTS=${FETCH_RESULTS:-1}
LOCAL_RESULTS_ROOT=${LOCAL_RESULTS_ROOT:-results/prime_runs}
SSH_OPTIONS=${SSH_OPTIONS:-"-o StrictHostKeyChecking=accept-new"}
REMOTE_CLEAN=${REMOTE_CLEAN:-0}

case "$MODE" in
  smoke|gpu-suite|lighteval-sweep|population-lighteval) ;;
  *)
    echo "MODE must be smoke, gpu-suite, lighteval-sweep, or population-lighteval, got $MODE" >&2
    exit 2
    ;;
esac

tmp_bundle=""
if [[ -n "$LOCAL_BUNDLE" ]]; then
  bundle="$LOCAL_BUNDLE"
else
  tmp_bundle=$(mktemp -t optimus-prime.XXXXXX.tar.gz)
  bundle="$tmp_bundle"
  COPYFILE_DISABLE=1 tar \
    --exclude='.git' \
    --exclude='.opencode' \
    --exclude='.pytest_cache' \
    --exclude='__pycache__' \
    --exclude='._*' \
    --exclude='.DS_Store' \
    --exclude='*.pyc' \
    --exclude='*.egg-info' \
    --exclude='external' \
    --exclude='data' \
    --exclude='docs/reports' \
    --exclude='logs' \
    --exclude='results' \
    --exclude='*.safetensors' \
    --exclude='*.pt' \
    --exclude='*.pth' \
    -czf "$bundle" .
fi

cleanup() {
  if [[ -n "$tmp_bundle" ]]; then
    rm -f "$tmp_bundle"
  fi
}
trap cleanup EXIT

if [[ "$REMOTE_CLEAN" == "1" ]]; then
  ssh $SSH_OPTIONS "$SSH_TARGET" "rm -rf '$REMOTE_ROOT' && mkdir -p '$REMOTE_ROOT'"
else
  ssh $SSH_OPTIONS "$SSH_TARGET" "mkdir -p '$REMOTE_ROOT'"
fi
scp $SSH_OPTIONS "$bundle" "$SSH_TARGET:$REMOTE_ROOT/source.tar.gz"
ssh $SSH_OPTIONS "$SSH_TARGET" "cd '$REMOTE_ROOT' && tar -xzf source.tar.gz && bash scripts/remote/optimus_prime_bootstrap.sh"

if [[ "$MODE" == "smoke" ]]; then
  ssh $SSH_OPTIONS "$SSH_TARGET" "cd '$REMOTE_ROOT' && TENSOR_PARALLEL_SIZE='$TENSOR_PARALLEL_SIZE' bash scripts/remote/optimus_prime_smoke.sh"
elif [[ "$MODE" == "gpu-suite" ]]; then
  ssh $SSH_OPTIONS "$SSH_TARGET" "cd '$REMOTE_ROOT' && TENSOR_PARALLEL_SIZE='$TENSOR_PARALLEL_SIZE' POPULATIONS='$POPULATIONS' PROMPT_VARIANTS='$PROMPT_VARIANTS' PROMPT_INPUT='$PROMPT_INPUT' USE_CHAT_TEMPLATE='$USE_CHAT_TEMPLATE' MAX_BASE_MALFORMED_FOR_SELECTION='$MAX_BASE_MALFORMED_FOR_SELECTION' MAX_BASE_CAP_HIT_FOR_SELECTION='$MAX_BASE_CAP_HIT_FOR_SELECTION' MIN_SELECTION_PROMPT_VARIANTS='$MIN_SELECTION_PROMPT_VARIANTS' REQUIRE_ALL_PROMPT_VARIANTS_VALID='$REQUIRE_ALL_PROMPT_VARIANTS_VALID' bash scripts/remote/optimus_prime_gpu_suite.sh"
elif [[ "$MODE" == "lighteval-sweep" ]]; then
  ssh $SSH_OPTIONS "$SSH_TARGET" "cd '$REMOTE_ROOT' && TENSOR_PARALLEL_SIZE='$TENSOR_PARALLEL_SIZE' DATA_PARALLEL_SIZE='$DATA_PARALLEL_SIZE' POPULATIONS='$POPULATIONS' MODEL='$MODEL' MODEL_TEMPLATE='$MODEL_TEMPLATE' TASKS='$TASKS' bash scripts/remote/optimus_prime_lighteval_sweep.sh"
else
  ssh $SSH_OPTIONS "$SSH_TARGET" "cd '$REMOTE_ROOT' && TENSOR_PARALLEL_SIZE='$TENSOR_PARALLEL_SIZE' DATA_PARALLEL_SIZE='$DATA_PARALLEL_SIZE' POPULATIONS='$POPULATIONS' MODEL='$MODEL' TASKS='$TASKS' PROMPT_VARIANTS='$PROMPT_VARIANTS' PROMPT_INPUT='$PROMPT_INPUT' USE_CHAT_TEMPLATE='$USE_CHAT_TEMPLATE' MAX_BASE_MALFORMED_FOR_SELECTION='$MAX_BASE_MALFORMED_FOR_SELECTION' MAX_BASE_CAP_HIT_FOR_SELECTION='$MAX_BASE_CAP_HIT_FOR_SELECTION' MIN_SELECTION_PROMPT_VARIANTS='$MIN_SELECTION_PROMPT_VARIANTS' REQUIRE_ALL_PROMPT_VARIANTS_VALID='$REQUIRE_ALL_PROMPT_VARIANTS_VALID' LIGHTEVAL_USE_CHAT_TEMPLATE='$LIGHTEVAL_USE_CHAT_TEMPLATE' bash scripts/remote/optimus_prime_population_lighteval.sh"
fi

if [[ "$FETCH_RESULTS" == "1" ]]; then
  mkdir -p "$LOCAL_RESULTS_ROOT"
  rsync -az -e "ssh $SSH_OPTIONS" "$SSH_TARGET:$REMOTE_ROOT/results/" "$LOCAL_RESULTS_ROOT/results/"
fi
