# T07 Dynamic CP C0/C1 Control Audit

## Scope

This note freezes what the pinned Megatron revision can actually provide to the
T07 control harness. It does not report a Dynamic CP performance result.

## Pinned implementation boundary

Megatron calls this feature **Hybrid Context Parallelism**. Its runtime is not
an attention-kernel flag. It contains four separate pieces:

1. `HybridCPDataLoaderWrapper` gathers variable sample lengths over the DPxCP
   group and reroutes samples with `all_to_all_single`.
2. `BalancedCPScheduler` chooses a power-of-two local CP size from the sample
   length and `max_seqlen_per_dp_cp_rank`.
3. `create_hybrid_dp_cp_groups` creates the reusable size-2/size-4 subgroups.
4. `hybrid_context_parallel_forward_backward` schedules the resulting work,
   switches local groups between samples, and inserts group-transition barriers.

The selected subgroup is passed into batch partitioning through
`local_cp_size` and `hybrid_cp_group`. Transformer Engine then updates the
DotProductAttention CP communicator from `PackedSeqParams.cp_group`.

## Required controls

| ID | Scheduler | Attention | Purpose |
| --- | --- | --- | --- |
| C0 | Static CP | Megatron standard SelfAttention | Variable-length reference with one fixed CP size |
| C1 | Hybrid/Dynamic CP | The same standard SelfAttention and weights | Isolate scheduler, reroute, and subgroup-sizing value |

C0 and C1 must use the same logical samples, parameters, local-LM-loss policy,
and optimizer-zero boundary. C1 headline timing must include sample-length
gather, schedule construction, sample reroute, subgroup selection, attention,
backward, and group-transition synchronization.

## Why this is separate from the MLA adapter

The pinned Megatron `MLASelfAttention` and absorbed MLA implementations
explicitly reject packed metadata with `local_cp_size`. Therefore C1 cannot be
implemented by enabling a flag on the current Native MLA adapter. Substituting
MLA into C1 would fail; silently falling back from Dynamic CP to static CP would
invalidate the control.

The existing benchmark runner also binds one case to one fixed CP group. A real
C1 iteration can contain several samples using different size-1/size-2/size-4
groups concurrently. It therefore needs a dedicated control-harness lifecycle
above the ordinary attention adapter:

```text
variable-length sample set
  -> DPxCP length gather
  -> BalancedCPScheduler
  -> all-to-all sample reroute
  -> per-round local CP subgroup
  -> standard SelfAttention forward/backward
  -> DPxCP gradient/loss reduction
```

Until that lifecycle is implemented and run, C1 must remain an explicit
capability gap and must not be represented by a static packed-THD MLA number.
