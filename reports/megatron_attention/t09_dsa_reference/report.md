# T09: Conservative MLA+DSA Integration Skeleton

## Outcome

T09 adds the first MagiAttention-side DSA integration seam while retaining the
pinned Megatron implementation as the exact correctness oracle. The adapter
replaces only the `DSAttention` class in Megatron's existing `ModuleSpec` with
`ConservativeMagiDSAttention`, a parameter-free subclass that delegates the
complete forward path to `DSAttention.forward`.

The skeleton passes CP1, CP2, CP4, and packed-varlen THD CP2 forward/backward
golden parity. It is intentionally not a sparse-communication optimization.

## Ownership boundary

Megatron continues to own:

- absorbed MLA Q/KV projections and RoPE;
- the DSA indexer and `megatron_dsa_v1` score semantics;
- exact top-k selection;
- document-local/causal validity masking;
- indexer loss and its gradient injection;
- CP AllGather of latent KV and indexer K;
- sparse attention and the absorbed V-up projection.

MagiAttention currently owns only the replaceable DSA backend class and
integration metadata. Every run therefore reports:

```text
communication_takeover = false
implementation = megatron_exact_allgather_reference
topk_implementation = megatron_exact_reference
```

This is deliberate: the next optimized backend must cross the same golden gate
before it can report `communication_takeover=true`.

## Golden matrix

All cases use one real Megatron absorbed-MLA/DSA layer, BF16, top-k 32, local LM
loss, and the same initial parameters and batch. Golden comparison includes the
canonical output, indexer loss, input gradient, and every parameter gradient.

| Layout | CP | Native | Magi seam | Output | Indexer loss | Input grad | Parameter grads |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| SBHD | 1 | pass | pass | max abs 0 | max abs 0 | max abs 0 | max abs 0 |
| SBHD | 2 | pass | pass | max abs 0 | max abs 0 | max abs 0 | max abs 0 |
| SBHD | 4 | pass | pass | max abs 0 | max abs 0 | max abs 0 | max abs 0 |
| packed THD | 2 | pass | pass | max abs 0 | max abs 0 | max abs 0 | max abs 0 |

For all four pairs:

- `allclose=true` at `atol=rtol=2e-3`;
- state-dict key sets are identical;
- parameter counts are identical (`1,313,280`);
- 24 checkpoint state keys are retained;
- the adapter records absorbed MLA and the expected CP size;
- no benchmark/GPU workers remain after completion.

The packed case contains document lengths `[256, 128, 64, 64]` and uses the
same per-document CP partitioning and document-local DSA mask semantics as the
Megatron oracle.

## What this proves

- MLA and DSA are integrated as one absorbed-MLA attention architecture rather
  than competing modes.
- A MagiAttention backend can enter through Megatron `ModuleSpec` without
  copying the complete attention module or changing its parameter/checkpoint
  surface.
- Indexer loss and indexer parameter gradients are part of the golden contract,
  not an untested auxiliary path.
- The conservative boundary works for CP1/2/4 and packed-varlen THD.

## What remains

- expose and compare exact top-k global IDs directly; current equality follows
  from delegating the identical pinned Megatron implementation;
- replace dense KV/indexer-K AllGather with a real Magi distributed exact-top-k
  and routed sparse-attention backend;
- add communication-byte and collective counters;
- validate TP, PP, recompute, checkpoint resume, and multi-layer index sharing;
- extend the same runner to one-layer training and the 3:1 Hybrid architecture.

Raw generated evidence lives in this directory as paired JSON reports, golden
tensor files, and `*_parity.json`. Those generated artifacts are kept outside
the source commit; this document is the durable summary.
