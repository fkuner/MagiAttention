#!/usr/bin/env bash
# Profile only the measured iterations of one benchmark case with Nsight Systems.

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${1:?usage: profile_case.sh CONFIG NPROC OUTPUT PROFILE_PREFIX}"
NPROC="${2:?usage: profile_case.sh CONFIG NPROC OUTPUT PROFILE_PREFIX}"
OUTPUT="${3:?usage: profile_case.sh CONFIG NPROC OUTPUT PROFILE_PREFIX}"
PROFILE_PREFIX="${4:?usage: profile_case.sh CONFIG NPROC OUTPUT PROFILE_PREFIX}"
LOG="${PROFILE_PREFIX}.log"
STATUS="${PROFILE_PREFIX}.status"

cd "${PROJECT_ROOT}"
mkdir -p "$(dirname "${OUTPUT}")" "$(dirname "${PROFILE_PREFIX}")"

export MAGI_BENCH_CUDA_PROFILER_RANGE=1
nsys profile \
  --trace=cuda,nvtx,cublas,cudnn,osrt \
  --sample=none \
  --cpuctxsw=none \
  --trace-fork-before-exec=true \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --force-overwrite=true \
  --output="${PROFILE_PREFIX}" \
  .venv/bin/torchrun \
    --standalone \
    --nproc-per-node="${NPROC}" \
    -m exps.megatron_attention.runner \
    --config "${CONFIG}" \
    --adapter native \
    --output "${OUTPUT}" \
  >"${LOG}" 2>&1
rc=$?
printf '%s\n' "${rc}" >"${STATUS}"

if [[ "${rc}" -eq 0 && -f "${PROFILE_PREFIX}.nsys-rep" ]]; then
  nsys stats \
    --report cuda_gpu_kern_sum,nvtx_gpu_proj_sum \
    --format csv \
    --output "${PROFILE_PREFIX}.stats" \
    "${PROFILE_PREFIX}.nsys-rep" \
    >>"${LOG}" 2>&1
fi

exit "${rc}"
