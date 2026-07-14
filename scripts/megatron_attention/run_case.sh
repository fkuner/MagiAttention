#!/usr/bin/env bash
# Run one benchmark case and persist diagnostics even when Python exits early.

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${1:?usage: run_case.sh CONFIG NPROC OUTPUT [ADAPTER]}"
NPROC="${2:?usage: run_case.sh CONFIG NPROC OUTPUT [ADAPTER]}"
OUTPUT="${3:?usage: run_case.sh CONFIG NPROC OUTPUT [ADAPTER]}"
ADAPTER="${4:-native}"
LOG="${OUTPUT%.json}.log"
STATUS="${OUTPUT%.json}.status"

cd "${PROJECT_ROOT}"
mkdir -p "$(dirname "${OUTPUT}")"

scripts/megatron_attention/run_in_env.sh \
  .venv/bin/torchrun \
  --standalone \
  --nproc-per-node="${NPROC}" \
  -m exps.megatron_attention.runner \
  --config "${CONFIG}" \
  --adapter "${ADAPTER}" \
  --output "${OUTPUT}" \
  >"${LOG}" 2>&1
rc=$?
printf '%s\n' "${rc}" >"${STATUS}"
exit "${rc}"
