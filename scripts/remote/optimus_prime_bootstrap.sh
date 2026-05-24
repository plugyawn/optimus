#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=${DEBIAN_FRONTEND:-noninteractive}
export PYTHONUNBUFFERED=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-ERROR}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export HF_HOME=${HF_HOME:-/root/hf_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME}
if [[ -n "${OPTIMUS_VLLM_ATTENTION_BACKEND:-}" ]]; then
  export VLLM_ATTENTION_BACKEND="$OPTIMUS_VLLM_ATTENTION_BACKEND"
fi

apt_install_if_available() {
  if command -v apt-get >/dev/null 2>&1; then
    for attempt in 1 2 3; do
      apt_get update && apt_get install -y --no-install-recommends \
        python3-venv python3-dev build-essential g++ g++-11 g++-12 git curl ca-certificates rsync tmux htop \
        && return 0
      sleep $((attempt * 10))
    done
    return 1
  fi
}

apt_get() {
  if [[ "$(id -u)" == "0" ]]; then
    env DEBIAN_FRONTEND="$DEBIAN_FRONTEND" apt-get "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo -E env DEBIAN_FRONTEND="$DEBIAN_FRONTEND" apt-get "$@"
  else
    return 1
  fi
}

apt_install_if_available || true

install_cuda_compiler_if_available() {
  if ! command -v apt-cache >/dev/null 2>&1; then
    return 0
  fi
  if ! command -v nvcc >/dev/null 2>&1; then
    for pkg in cuda-compiler-13-0 cuda-compiler-12-2 cuda-nvcc-13-0 cuda-nvcc-12-2; do
      if apt-cache show "$pkg" >/dev/null 2>&1; then
        apt_get install -y --no-install-recommends "$pkg" || return 1
        break
      fi
    done
  fi
  for pkg in libcurand-dev-13-0 libcurand-dev-12-2 cuda-libraries-dev-13-0 cuda-libraries-dev-12-2; do
    if apt-cache show "$pkg" >/dev/null 2>&1; then
      apt_get install -y --no-install-recommends "$pkg" || true
      break
    fi
  done
  if [[ ! -e /usr/local/cuda ]]; then
    for cuda_dir in /usr/local/cuda-13.0 /usr/local/cuda-12.2 /usr/local/cuda-*; do
      if [[ -d "$cuda_dir" ]]; then
        if [[ "$(id -u)" == "0" ]]; then
          ln -s "$cuda_dir" /usr/local/cuda
        elif command -v sudo >/dev/null 2>&1; then
          sudo ln -s "$cuda_dir" /usr/local/cuda
        fi
        break
      fi
    done
  fi
}

install_cuda_compiler_if_available || true

python_bin=${PYTHON_BIN:-python3}
"$python_bin" -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel

if [[ "${INSTALL_VLLM:-1}" == "1" ]]; then
  python -m pip install --upgrade "${OPTIMUS_VLLM_PACKAGE:-vllm>=0.19.0}"
fi

if [[ "${OPTIMUS_INSTALL_FLASHINFER:-1}" == "1" ]]; then
  if ! python -m pip install --upgrade "${OPTIMUS_FLASHINFER_PACKAGE:-flashinfer-python}"; then
    if [[ "${OPTIMUS_REQUIRE_FLASHINFER:-0}" == "1" ]]; then
      exit 1
    fi
    printf '%s\n' "warning: flashinfer-python install failed; continuing with vLLM automatic backend selection."
  fi
fi

if [[ "${OPTIMUS_INSTALL_FLASH_ATTN:-0}" == "1" ]]; then
  if ! python -m pip install --upgrade "${OPTIMUS_FLASH_ATTN_PACKAGE:-flash-attn}" --no-build-isolation; then
    if [[ "${OPTIMUS_REQUIRE_FLASH_ATTN:-0}" == "1" ]]; then
      exit 1
    fi
    printf '%s\n' "warning: flash-attn install failed; continuing with available vLLM kernels."
  fi
fi

python -m pip install -e ".[dev]"
python -m pip install --upgrade "${OPTIMUS_TRANSFORMERS_PACKAGE:-transformers>=4.51,<5}"
optimus --help >/dev/null
find optimus -name '._*' -delete
python -m compileall -q optimus

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi
