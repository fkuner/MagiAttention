# T06 Megatron-native scaffold report

Status: **PASS**

This gate proves that the shared runner can execute real modules from the pinned
Megatron checkout without importing MagiAttention in the native adapter. It is
a structural/correctness gate, not a performance result.

## Frozen provenance

- MagiAttention active checkout: `magic@529fb0a4e273b3557a56d8afd60b74da46688095`
- Megatron baseline: `779c5b748dbcf00ad9e36d539c576b404ab4abe9`
- Remote project: `/root/work/MagiAttention`
- Interpreter: `/root/work/MagiAttention/.venv/bin/python`
- Runtime: Python 3.12.13, PyTorch 2.9.1+cu130, Transformer Engine 2.16.0
- Hardware identity: the driver reports `NVIDIA L20C`; the user confirmed B200,
  while CUDA independently reports compute capability 10.0 (SM100), 148 SMs and
  191503138816 bytes per GPU. The raw label is retained rather than rewritten.

The selected package additions and the process audit are recorded in
`environment_addendum.json` and `process_audit.txt` in this directory.

## Final native cases

| Artifact | Harness | Topology/layout | Result | Regression evidence |
| --- | --- | --- | --- | --- |
| `reports/t06_native_mla_cp1_final.json` | attention-layer E2E | CP1 SBHD dense MLA | PASS | non-contiguous backward; 6 finite parameter gradients; state clone; 3 QKV observations |
| `reports/t06_native_mla_loss_cp1_r3.json` | one-layer-training E2E | CP1 SBHD dense MLA + tied local LM loss | PASS | non-contiguous backward; 6 finite parameter gradients; state clone; 3 QKV observations |
| `reports/t06_native_dsa_cp1_final.json` | attention-layer E2E | CP1 SBHD absorbed MLA+DSA | PASS | non-contiguous backward; 13 finite parameter gradients; state clone; 3 QKV observations |
| `reports/t06_native_hybrid_cp1_final.json` | one-layer-training E2E | CP1 `3xGDN + 1xMLA` | PASS | non-contiguous backward; 27 finite parameter gradients; state clone; 3 QKV observations |
| `reports/t06_native_packed_cp2_final.json` | attention-layer E2E | rank 0, CP2 THD packed-varlen | PASS | per-document mapping; non-contiguous backward; 6 finite gradients; state clone; 3 QKV observations |
| `reports/t06_native_packed_cp2_final.rank1.json` | attention-layer E2E | rank 1, CP2 THD packed-varlen | PASS | complementary per-document mapping; same regression gates |

All cases run real Megatron `MLASelfAttention`, absorbed DSA, or
`GatedDeltaNet`; the adapter does not copy their forward implementations. The
explicit lifecycle is `prepare -> module forward -> finalize`, and Native/Magi
state sharing is represented by an independent CPU-cloned state dictionary.

The focused integration suite completed with `33 passed` in 17.70 seconds in
the project environment. Isort checks, compile checks, and `git diff --check`
are part of this gate's final validation.

## Resolved capability findings

1. Official `flash-attn-4==4.0.0b11` is installed and detected by Transformer
   Engine. This package is distinct from PR #331's `flash-attn-cute` source.
2. Transformer Engine 2.16 explicitly rejects FA4 with context parallelism.
   Therefore official FA4 is not used to claim a native CP result.
3. Transformer Engine 2.16 does not expose a valid `THD + CP + all_gather`
   backend for this dense MLA shape. The final packed dense MLA CP2 control uses
   the supported `p2p` path. This does not change the separate DSA AllGather
   oracle required by T07/T10.
4. The optional fast-Hadamard package is absent. DSA indexer activation
   rotation is explicitly disabled; applying the same orthogonal rotation to Q
   and K preserves their scores and exact top-k semantics. The result JSON
   records `dsa_indexer_rotate_activation=false`.
5. The one-iteration timing fields are smoke/JIT diagnostics only. They must not
   be compared as performance data; T07 freezes native baselines and T09 runs
   the measured E2E matrix.

## Scope boundary

The Hybrid case only proves the scaffold can construct and train a
`GDN -> GDN -> GDN -> MLA` unit. It does not complete the T14 Hybrid milestone,
does not establish persistent Magi layout across recurrent layers, and does not
claim any speedup.

No Megatron baseline source file was modified for T06. The four pre-existing
user-modified MagiAttention files retain their T00 SHA-256 values.
