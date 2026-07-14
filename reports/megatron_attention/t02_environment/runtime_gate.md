# T02 remote environment and PR #331 runtime Gate

Status: **PASS**

The only benchmark interpreter is `/root/work/MagiAttention/.venv/bin/python`.
It imports MagiAttention from the clean PR #331 worktree and Megatron from the
pinned clean baseline checkout. The original `/opt/conda` environment and the
older LongAttention environment were not modified.

## Verified cases

| Case | Result |
| --- | --- |
| BF16 CUDA matmul | PASS |
| NCCL CP2 all-reduce + all-gather | PASS |
| NCCL CP4 all-reduce + all-gather | PASS |
| PR #331 FA4, CP1, BF16, GQA 8:2, Q/K=192, V=128, 12K varlen, forward/backward | PASS |
| PR #331 FA4, CP4, BF16, GQA 8:2, Q/K=192, V=128, 12K varlen, forward/backward | PASS |
| Identical CP4 correctness repeat | PASS |

The CP4 test compared output, LSE, dQ, dK and dV with the PR's reference path.
The first CP4 run completed in 89.143 seconds and the identical repeat completed
in 116.233 seconds. These are correctness-run wall times, not benchmark numbers:
the test includes layout solving, process startup, communication, high- and
low-precision reference attention, and elementwise error analysis. T06 must
measure cold compilation and steady execution with the dedicated E2E runner.

## Resolved test harness issue

The first attempt exited before the requested parameter filter because the
container inherited `MAGI_ATTENTION_SANITY_CHECK=0`, while non-profile pipeline
tests assert that sanity checking is enabled before applying case filters. It
was stopped gracefully with SIGINT and rerun with
`MAGI_ATTENTION_SANITY_CHECK=1`. No target FA4 kernel ran in the failed attempt.

## GPU identity

The driver reports `NVIDIA L20C`. The user confirmed the hardware is B200, and
the runtime independently reports compute capability 10.0 (SM100), 148 SMs and
191503138816 bytes per GPU. Both facts are retained; the raw driver label is not
silently rewritten.
