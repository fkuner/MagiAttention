# T07 Megatron Native Matrix

> This is a correctness and capability matrix. Its 512-token timings are not headline performance results.

- Status: `pass`
- Cases: `21`
- Status counts: `{"pass": 15, "unsupported_or_failed": 6}`
- Missing required groups: `[]`

## Selected Native baselines

| Group | Case | CP mode | Median E2E (ms) | Passing candidates |
| --- | --- | --- | ---: | ---: |
| `dense_sbhd_cp1` | `t07_probe_dense_sbhd_cp1_none` | `none` | 4.746525 | 1 |
| `dense_sbhd_cp2` | `t07_probe_dense_sbhd_cp2_p2p` | `p2p` | 5.715593 | 3 |
| `dense_sbhd_cp4` | `t07_probe_dense_sbhd_cp4_all_gather` | `all_gather` | 5.713786 | 3 |
| `dense_thd_cp1` | `t07_probe_dense_thd_cp1_none` | `none` | 6.435027 | 1 |
| `dense_thd_cp2` | `t07_probe_dense_thd_cp2_p2p` | `p2p` | 8.222470 | 2 |
| `dense_thd_cp4` | `t07_probe_dense_thd_cp4_p2p` | `p2p` | 10.967083 | 2 |
| `dsa_sbhd_cp1` | `t07_probe_dsa_sbhd_cp1_none` | `none` | 8.834623 | 1 |
| `dsa_sbhd_cp2` | `t07_probe_dsa_sbhd_cp2_allgather` | `allgather` | 11.259479 | 1 |
| `dsa_sbhd_cp4` | `t07_probe_dsa_sbhd_cp4_allgather` | `allgather` | 10.648319 | 1 |

## All cases

| Case | Group | CP mode | Status | Repeat match | Median E2E (ms) | Max CV |
| --- | --- | --- | --- | --- | ---: | ---: |
| `t07_probe_dense_sbhd_cp1_none` | `dense_sbhd_cp1` | `none` | `pass` | `True` | 4.746525 | 0.257853 |
| `t07_probe_dense_sbhd_cp2_p2p` | `dense_sbhd_cp2` | `p2p` | `pass` | `True` | 5.715593 | 0.516190 |
| `t07_probe_dense_sbhd_cp2_all_gather` | `dense_sbhd_cp2` | `all_gather` | `pass` | `True` | 6.241607 | 0.015310 |
| `t07_probe_dense_sbhd_cp2_a2a` | `dense_sbhd_cp2` | `a2a` | `pass` | `True` | 6.517879 | 0.064578 |
| `t07_probe_dense_sbhd_cp2_hierarchical` | `dense_sbhd_cp2` | `hierarchical` | `unsupported_or_failed` | n/a | n/a | n/a |
| `t07_probe_dense_sbhd_cp4_p2p` | `dense_sbhd_cp4` | `p2p` | `pass` | `True` | 7.713853 | 0.151309 |
| `t07_probe_dense_sbhd_cp4_all_gather` | `dense_sbhd_cp4` | `all_gather` | `pass` | `True` | 5.713786 | 0.088393 |
| `t07_probe_dense_sbhd_cp4_a2a` | `dense_sbhd_cp4` | `a2a` | `pass` | `True` | 8.456478 | 0.494543 |
| `t07_probe_dense_sbhd_cp4_hierarchical` | `dense_sbhd_cp4` | `hierarchical` | `unsupported_or_failed` | n/a | n/a | n/a |
| `t07_probe_dense_thd_cp1_none` | `dense_thd_cp1` | `none` | `pass` | `True` | 6.435027 | 0.032594 |
| `t07_probe_dense_thd_cp2_p2p` | `dense_thd_cp2` | `p2p` | `pass` | `True` | 8.222470 | 0.035949 |
| `t07_probe_dense_thd_cp2_all_gather` | `dense_thd_cp2` | `all_gather` | `unsupported_or_failed` | n/a | n/a | n/a |
| `t07_probe_dense_thd_cp2_a2a` | `dense_thd_cp2` | `a2a` | `pass` | `True` | 17.794429 | 0.014279 |
| `t07_probe_dense_thd_cp2_a2a+p2p` | `dense_thd_cp2` | `a2a+p2p` | `unsupported_or_failed` | n/a | n/a | n/a |
| `t07_probe_dense_thd_cp4_p2p` | `dense_thd_cp4` | `p2p` | `pass` | `True` | 10.967083 | 0.283099 |
| `t07_probe_dense_thd_cp4_all_gather` | `dense_thd_cp4` | `all_gather` | `unsupported_or_failed` | n/a | n/a | n/a |
| `t07_probe_dense_thd_cp4_a2a` | `dense_thd_cp4` | `a2a` | `pass` | `True` | 32.661215 | 0.163163 |
| `t07_probe_dense_thd_cp4_a2a+p2p` | `dense_thd_cp4` | `a2a+p2p` | `unsupported_or_failed` | n/a | n/a | n/a |
| `t07_probe_dsa_sbhd_cp1_none` | `dsa_sbhd_cp1` | `none` | `pass` | `True` | 8.834623 | 0.020622 |
| `t07_probe_dsa_sbhd_cp2_allgather` | `dsa_sbhd_cp2` | `allgather` | `pass` | `True` | 11.259479 | 0.254338 |
| `t07_probe_dsa_sbhd_cp4_allgather` | `dsa_sbhd_cp4` | `allgather` | `pass` | `True` | 10.648319 | 0.251362 |

## Sources

- `reports/megatron_attention/t07_native_matrix/probe_fa2_full_summary_v2.json`
- `reports/megatron_attention/t07_native_matrix/probe_fa2_dsa_resume_summary_v3.json`
