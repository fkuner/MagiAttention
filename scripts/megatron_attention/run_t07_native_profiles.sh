#!/usr/bin/env bash
# Capture the selected CP4 Native attention-stack baselines sequentially.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/reports/megatron_attention/t07_native_matrix/nsys"

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"

scripts/megatron_attention/profile_case.sh \
  exps/megatron_attention/cases/t07_probe/t07_probe_dense_sbhd_cp4_all_gather.json \
  4 \
  "${OUTPUT_DIR}/dense_sbhd_cp4_all_gather.json" \
  "${OUTPUT_DIR}/dense_sbhd_cp4_all_gather"

scripts/megatron_attention/profile_case.sh \
  exps/megatron_attention/cases/t07_probe/t07_probe_dense_thd_cp4_p2p.json \
  4 \
  "${OUTPUT_DIR}/dense_thd_cp4_p2p.json" \
  "${OUTPUT_DIR}/dense_thd_cp4_p2p"

scripts/megatron_attention/profile_case.sh \
  exps/megatron_attention/cases/t07_probe/t07_probe_dsa_sbhd_cp4_allgather.json \
  4 \
  "${OUTPUT_DIR}/dsa_sbhd_cp4_allgather.json" \
  "${OUTPUT_DIR}/dsa_sbhd_cp4_allgather"
