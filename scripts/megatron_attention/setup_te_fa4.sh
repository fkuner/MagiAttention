#!/usr/bin/env bash
# Install/check the FA2+FA4 packages required by Transformer Engine CP.

set -euo pipefail

MODE="${1:-check}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_ROOT="${MAGI_BENCHMARK_VENV:-${PROJECT_ROOT}/.venv}"
INDEX_URL="${PIP_INDEX_URL:-https://pypi.antfin-inc.com/simple/}"
FA2_VERSION="2.8.3"
FA4_VERSION="4.0.0b11"

case "${MODE}" in
  dry-run)
    uv pip install \
      --dry-run \
      --no-deps \
      --python "${VENV_ROOT}/bin/python" \
      --index-url "${INDEX_URL}" \
      "flash-attn-4==${FA4_VERSION}"
    ;;
  install)
    uv pip install \
      --no-deps \
      --python "${VENV_ROOT}/bin/python" \
      --index-url "${INDEX_URL}" \
      "flash-attn-4==${FA4_VERSION}"
    ;;
  install-stack)
    # TE 2.16 accepts FA2 <=2.8.3. FA2 and FA4 share the flash_attn
    # namespace, so FA4 must be installed last to restore its cute modules.
    PIP_INDEX_URL="${INDEX_URL}" \
      FLASH_ATTN_CUDA_ARCHS=100 \
      MAX_JOBS="${MAX_JOBS:-8}" \
      FLASH_ATTENTION_FORCE_BUILD=TRUE \
      "${VENV_ROOT}/bin/python" -m pip install \
      "flash-attn==${FA2_VERSION}" --no-build-isolation --no-deps
    PIP_INDEX_URL="${INDEX_URL}" \
      "${VENV_ROOT}/bin/python" -m pip install \
      --force-reinstall "flash-attn-4==${FA4_VERSION}" --no-deps
    ;;
  check)
    "${PROJECT_ROOT}/scripts/megatron_attention/run_in_env.sh" \
      "${VENV_ROOT}/bin/python" -c \
      'import importlib.metadata as m, flash_attn, transformer_engine; from flash_attn.cute.interface import flash_attn_varlen_func; print(transformer_engine.__version__, flash_attn.__version__, m.version("flash-attn-4"), m.version("nvidia-cublas"), flash_attn_varlen_func.__module__)'
    ;;
  *)
    echo "usage: setup_te_fa4.sh {dry-run|install|install-stack|check}" >&2
    exit 2
    ;;
esac
