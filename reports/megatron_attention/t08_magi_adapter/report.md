# T08: Magi Expanded-MLA Adapter

## Outcome

T08 establishes a correctness-first Magi adapter around the real Megatron
expanded-MLA attention layer. Megatron retains ownership of MLA projections,
RoPE, output projection, parameters, and checkpoint structure. The adapter
replaces only `MLASelfAttention.core_attention` with a parameter-free Magi
backend based on the asymmetric Q/K and V dimensions from MagiAttention PR
#331.

This gate is complete for the tested dense MLA scope. It is not a performance
claim and it does not yet implement DSA.

## Integration boundary

The tested execution path is:

```text
Megatron hidden state and CP layout
  -> Megatron MLA down/up projections and RoPE
  -> reconstruct global expanded Q/K/V in physical token order
  -> Magi dispatch
  -> Magi calc_attn
  -> Magi undispatch
  -> restore Megatron CP-local token order
  -> Megatron output projection
```

The layout policy is `attention_local_dispatch_combine`. It deliberately keeps
the inter-layer hidden-state layout under Megatron ownership so future KDA/GDN
recurrent layers continue to observe ordered tokens. The current implementation
uses dense Q/K/V reconstruction before Magi dispatch; persistent Magi layout and
communication optimization are later gates.

## Golden parity

All comparisons use the same Megatron-created module state and input, BF16
execution, and `atol=rtol=2e-3`.

| Layout | CP | Output max abs | Input-grad max abs | Max parameter-grad abs | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| SBHD | 1 | 6.1035e-05 | 7.4506e-09 | 4.7684e-07 | pass |
| SBHD | 2 | 6.1035e-05 | 7.9721e-07 | 3.1948e-05 | pass |
| SBHD | 4 | 6.1035e-05 | 1.3262e-06 | 4.2915e-05 | pass |
| packed THD | 2 | 1.2207e-04 | 1.7025e-06 | 7.5340e-05 | pass |

The four Magi runs report `communication_takeover=true`. For every pair, the
state-dict keys, parameter count (`1,179,648`), and parameter-state hashes match
the Megatron Native oracle.

Raw evidence is generated under this directory as `native_*.json`,
`magi_*.json`, `*_parity.json`, and `*.golden.pt`. These generated artifacts are
kept out of the source commit; this report is the durable summary.

## Scope and remaining gates

Proven here:

- real Megatron expanded-MLA projections and output projection;
- Magi PR #331 asymmetric Q/K (`192`) and V (`128`) attention dimensions;
- SBHD CP1/2/4 forward and backward parity;
- packed-varlen THD CP2 forward and backward parity;
- unchanged Megatron parameter and checkpoint surfaces;
- external backend selection without replacing Megatron's complete attention
  module.

Not proven here:

- DSA or exact top-k semantics;
- indexer loss and indexer gradients;
- persistent Magi token layout between layers;
- a latency, memory, or communication advantage;
- uneven Magi shards, pipeline parallelism, recompute, or checkpoint resume;
- the hybrid KDA/GDN training path.

The next gate is a conservative MLA+DSA skeleton that preserves Megatron's
exact AllGather behavior as the correctness oracle before Magi takes over any
sparse communication.
