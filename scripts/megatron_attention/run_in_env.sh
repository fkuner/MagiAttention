#!/usr/bin/env bash
# Run one command with the benchmark environment's wheel-provided CUDA libraries.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_ROOT="${MAGI_BENCHMARK_VENV:-${PROJECT_ROOT}/.venv}"
MEGATRON_ROOT="${MEGATRON_ROOT:-/root/work/megatron}"
MAGI_SOURCE_ROOT="${MAGI_SOURCE_ROOT:-${PROJECT_ROOT}}"
PYTHON_SITE="${VENV_ROOT}/lib/python3.12/site-packages"
CUDA_WHEEL_LIB="${PYTHON_SITE}/nvidia/cu13/lib"

if [[ ! -x "${VENV_ROOT}/bin/python" ]]; then
  echo "benchmark environment is missing: ${VENV_ROOT}/bin/python" >&2
  exit 2
fi

if [[ ! -d "${CUDA_WHEEL_LIB}" ]]; then
  echo "wheel-provided CUDA library directory is missing: ${CUDA_WHEEL_LIB}" >&2
  exit 2
fi

export LD_LIBRARY_PATH="${CUDA_WHEEL_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${MAGI_SOURCE_ROOT}:${MEGATRON_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

exec "$@"
