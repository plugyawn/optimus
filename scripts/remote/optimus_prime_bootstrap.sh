#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=${DEBIAN_FRONTEND:-noninteractive}
export PYTHONUNBUFFERED=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-ERROR}
export HF_HOME=${HF_HOME:-/root/hf_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME}

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
  python -m pip install --upgrade "${OPTIMUS_VLLM_PACKAGE:-vllm==0.9.2}"
fi

python -m pip install -e ".[dev]"
python -m pip install --upgrade "${OPTIMUS_TRANSFORMERS_PACKAGE:-transformers==4.51.3}"
optimus --help >/dev/null
compile_dirs=()
for source_dir in optimus randopt_lora_lab; do
  if [[ -d "$source_dir" ]]; then
    find "$source_dir" -name '._*' -delete
    compile_dirs+=("$source_dir")
  fi
done
python -m compileall -q "${compile_dirs[@]}"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi
