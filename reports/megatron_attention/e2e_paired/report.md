# Megatron Native vs Magi Attention E2E baseline

## Scope

This baseline compares a real Megatron attention stack with Magi adapters from
batch materialization onward. The timed region includes batch dispatch,
attention forward, loss construction, backward, and gradient zeroing. It is not
a kernel-only benchmark.

Two harnesses are kept separate:

- `attention_stack`: real Megatron MLA projections, RoPE, core attention, output
  projection, a deterministic fixed-gradient objective, backward, and zeroing.
- `attention_to_loss`: the same attention stack followed by tied-vocabulary
  logits, packed/local LM loss, backward, and zeroing.

Native and Magi runs use the same case schema, seed, initial state, logical
sequence, model dimensions, dtype, warmup, iterations, and process topology.
Launch order alternates by repeat. Performance is reported only after output and
gradient parity passes.

## Environment and provenance

- Remote checkout: `/root/work/MagiAttention`, branch `magic`.
- Megatron checkout: `/root/work/megatron`.
- Python environment: `/root/work/MagiAttention/.venv`.
- Hardware: 4 NVIDIA B200 GPUs. A previous `L20C` label was a driver reporting
  issue and is not used as hardware provenance.
- MLA source: MagiAttention PR #331, with asymmetric QK/V dimensions enabled.
- DSA source of truth: Megatron's exact AllGather implementation. The current
  Magi DSA adapter is a conservative integration seam and does not take over
  DSA communication.

## Correctness gates

The paired comparator uses separate gates:

- output and optional semantic tensors: `atol=1e-3`, `rtol=1e-3`;
- input and parameter gradients: `atol=2e-3`, `rtol=2e-3`;
- identical parameter-gradient key sets and optional-tensor key sets.

The matrix includes dense MLA and conservative DSA, SBHD CP1/2/4, packed THD
CP1/2, and both harnesses. Each pair runs twice with reversed launch order.

## Initial 16-pair matrix

The first complete run covered SBHD CP1/2/4 and THD CP2. Fifteen pairs passed.
The only failure was dense THD CP2 `attention_to_loss`:

- output max absolute difference: `1.2207e-4`;
- input-gradient max absolute difference: `3.1781e-4`;
- KV-down gradient max absolute difference: `5.0659e-3`;
- KV-up gradient max absolute difference: `5.5342e-3`.

The failure was deterministic in both launch orders. It was preserved rather
than hidden by increasing tolerances.

Representative pre-fix timings are below. `Native / Magi` smaller than one
means the current correctness-first Magi adapter is slower.

| Pair | Native median (ms) | Magi median (ms) | Native / Magi |
| --- | ---: | ---: | ---: |
| dense SBHD CP1 attention-stack | 3.560 | 4.871 | 0.731 |
| dense SBHD CP2 attention-stack | 5.340 | 8.575 | 0.623 |
| dense SBHD CP4 attention-stack | 7.031 | 9.472 | 0.742 |
| dense THD CP2 attention-stack | 8.206 | 10.935 | 0.750 |
| dense SBHD CP2 attention-to-loss | 5.725 | 8.808 | 0.650 |
| dense SBHD CP4 attention-to-loss | 7.051 | 10.335 | 0.682 |

These numbers establish the golden baseline; they are not an optimization
claim. The adapter deliberately reconstructs global Q/K/V, performs Magi
dispatch/attention/undispatch, and restores Megatron's native layout at every
attention layer.

DSA timings are not a Native-vs-Magi algorithm comparison. Both adapters use
the same Megatron exact AllGather DSA reference (`communication_takeover=false`),
so ratios around one are launch noise.

## Packed local-loss diagnosis and fix

Two independent checks isolated the failing gradient path:

1. Dense THD CP1 passed both harnesses in both launch orders. Packed batch,
   labels, loss mask, logits, and local-loss semantics are therefore correct
   without distributed output routing.
2. CP1 Native versus CP2 Native passed, with maximum KV gradient difference
   `3.8147e-5`. CP1 Magi versus CP2 Magi failed with the same approximately
   `5.5e-3` KV-gradient error as the paired failure.

The adapter called Magi `undispatch` with its default backward. After
undispatch, each native CP rank consumes only its own token subset, so each rank
holds a partial contribution to the global output gradient. The default
undispatch backward only selects local Magi chunks and does not combine those
partial contributions.

The fix is to call:

```python
undispatch(magi_output, runtime_key, is_partial_grad=True)
```

Magi then uses reduce-scatter to sum and route partial global-output gradients
back to the correct Magi output owners. This is a semantic fix, not a tolerance
change.

After the fix, dense THD CP2 passed both harnesses in both launch orders. For
`attention_to_loss`, representative maxima became:

- output: `1.2207e-4`;
- input gradient: `1.9073e-6`;
- KV-down gradient: `3.8147e-5`;
- KV-up gradient: `6.1035e-5`.

Post-fix dense THD CP2 medians:

| Pair | Native median (ms) | Magi median (ms) | Native / Magi |
| --- | ---: | ---: | ---: |
| attention-stack | 8.064 | 11.936 | 0.676 |
| attention-to-loss | 8.263 | 10.951 | 0.755 |

The same gradient-routing fix was then regressed on dense SBHD without changing
the gates. CP2 and CP4 both passed both harnesses in both launch orders:

| Pair | Native median (ms) | Magi median (ms) | Native / Magi |
| --- | ---: | ---: | ---: |
| SBHD CP2 attention-stack | 5.571 | 8.906 | 0.626 |
| SBHD CP2 attention-to-loss | 6.319 | 9.127 | 0.692 |
| SBHD CP4 attention-stack | 6.559 | 9.294 | 0.706 |
| SBHD CP4 attention-to-loss | 6.950 | 10.844 | 0.641 |

The additional reduce-scatter is required by the current attention-local
layout bridge. A future persistent Magi layout can remove this bridge rather
than optimizing away a required gradient reduction.

## Current conclusions

- The benchmark now starts at the requested batch-generation/dispatch boundary
  and covers both attention-layer and one-layer-training E2E.
- The Native adapter calls Megatron's real attention modules and is a numerical
  oracle, not a reimplementation of MLA/DSA math.
- PR #331 expanded MLA works through real Megatron projections, RoPE, output
  projection, CP, asymmetric QK/V dimensions, and packed THD.
- Conservative DSA is integrated but still delegates exact DSA and dense CP
  communication to Megatron.
- The correctness-first Magi MLA adapter is currently slower than Native. This
  is expected and gives the project an honest optimization baseline.
- Persistent cross-layer layout and PP stage-local runtime context remain later
  gates. Hybrid KDA/GDN layers are not on the critical path until MLA+DSA and
  the integration skeleton are stable.

## Preserved evidence

Remote artifacts are stored under:

```text
/root/work/MagiAttention/reports/megatron_attention/e2e_paired/
```

Important summaries:

- `full_summary.json`: original complete 16-pair run, including the preserved
  pre-fix failure;
- `strict_summary.json` and `strict_summary_dsa_thd.json`: strict offline
  rechecks of preserved golden artifacts;
- `cp1_thd_oracle_summary.json`: CP1 packed oracle;
- `partial_grad_fix_summary.json`: post-fix THD CP2 rerun;
- `sbhd_partial_grad_regression_summary.json`: completed post-fix SBHD CP2 pair
  records (the combined launcher timed out before starting CP4, so its top-level
  status remains `running`);
- `sbhd_cp4_partial_grad_summary.json`: completed post-fix SBHD CP4 regression,
  with both pairs and all four launch-order repetitions passing.

Raw JSON, logs, and `.golden.pt` tensors remain untracked benchmark evidence;
this report is the durable, reviewable summary.
