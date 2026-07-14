"""Correctness-first Magi adapter around Megatron's real MLA layer."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

from .native import MegatronNativeAdapter, NativePreparedBatch


@dataclass
class MagiPreparedBatch(NativePreparedBatch):
    runtime_key: object | None = None


class MagiExpandedMLAAdapter(MegatronNativeAdapter):
    """Keep Megatron layout between layers and use Magi inside core attention."""

    @property
    def name(self) -> str:
        return "magi_expanded_mla"

    def build(self, case, process_groups, shared_state):
        if case.attention_mode != "dense_mla":
            raise NotImplementedError("T08 Magi adapter supports dense expanded MLA first")
        if case.architecture != "attention":
            raise NotImplementedError("T08 Magi adapter supports attention-only recipes first")
        os.environ.pop("MAGI_ATTENTION_SDPA_BACKEND", None)
        os.environ.pop("MAGI_ATTENTION_FA4_BACKEND", None)
        os.environ.setdefault("MAGI_ATTENTION_KERNEL_BACKEND", "fa4")
        return super().build(case, process_groups, shared_state)

    def customize_dense_submodules(self, dense_submodules, case):
        from magi_attention.integrations.megatron.expanded_mla import (
            MagiExpandedMLACoreAttention,
        )

        return replace(dense_submodules, core_attention=MagiExpandedMLACoreAttention)

    def dispatch_batch(self, packed_batch, plan, case):
        prepared = super().dispatch_batch(packed_batch, plan, case)
        from magi_attention.api import magi_attn_varlen_key
        from magi_attention.config import (
            DispatchConfig,
            DistAttnConfig,
            MinHeapDispatchAlg,
        )

        total_tokens = packed_batch.physical_num_tokens
        if sum(case.sequence.logical_lengths) != total_tokens:
            raise NotImplementedError("T08 requires logical lengths to equal physical lengths")
        chunk_size = min(512, min(case.sequence.logical_lengths))
        # Keep the first gate deterministic and evenly sharded.
        while chunk_size > 1 and total_tokens % (chunk_size * case.parallel.cp):
            chunk_size //= 2
        if total_tokens % (chunk_size * case.parallel.cp):
            raise ValueError("cannot form an even Magi dispatch for this T08 case")
        cu = packed_batch.cu_seqlens_q.to(device=self._device)
        local_heads = case.model.num_attention_heads // case.parallel.tp
        runtime_key = magi_attn_varlen_key(
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
            num_heads_q=local_heads,
            # Expanded MLA materializes one K/V head per query head.
            num_heads_kv=local_heads,
            head_dim=case.model.execution_qk_head_dim,
            head_dim_v=case.model.v_head_dim,
            pad_size=0,
            cp_group_or_mesh=self._pg.cp,
            causal=True,
            dist_attn_config=DistAttnConfig(
                dispatch_config=DispatchConfig(
                    chunk_size=chunk_size,
                    uneven_shard=False,
                    alg=MinHeapDispatchAlg(),
                )
            ),
        )
        return MagiPreparedBatch(**prepared.__dict__, runtime_key=runtime_key)

    def forward(self, modules, prepared_batch, case):
        from magi_attention.integrations.megatron.expanded_mla import ExpandedMLARuntime

        runtime = ExpandedMLARuntime(
            key=prepared_batch.runtime_key,
            local_to_global=prepared_batch.local_to_global,
            cp_group=self._pg.cp,
        )
        for layer in modules.layers:
            layer.core_attention.bind_runtime(runtime)
        return super().forward(modules, prepared_batch, case)

    def correctness_metadata(self, canonical, modules, prepared_batch, case):
        metadata = super().correctness_metadata(canonical, modules, prepared_batch, case)
        executions = [layer.core_attention.last_execution for layer in modules.layers]
        metadata.update(
            {
                "scope": "T08 Magi expanded MLA over real Megatron projections/RoPE/output",
                "communication_takeover": all(
                    execution.get("communication_takeover", False) for execution in executions
                ),
                "magi_executions": executions,
                "layout_policy": "attention_local_dispatch_combine",
            }
        )
        return metadata
