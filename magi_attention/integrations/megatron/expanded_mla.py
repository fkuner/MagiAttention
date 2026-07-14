"""Megatron core-attention adapter for Magi expanded MLA.

Megatron owns MLA projections, RoPE, and the output projection.  This module
only converts the already-projected Q/K/V tensors from Megatron's CP layout to
the Magi layout, executes distributed attention, and restores Megatron's
layout.  The conversion is intentionally correctness-first; persistent Magi
layout is a later optimization once ordered recurrent layers can consume it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn


@dataclass(frozen=True)
class ExpandedMLARuntime:
    """Per-microbatch state bound by the benchmark/integration adapter."""

    key: object
    local_to_global: torch.Tensor
    cp_group: object


class MagiExpandedMLACoreAttention(nn.Module):
    """Parameter-free Magi backend for Megatron's expanded MLA Q/K/V."""

    supports_mla_asymmetric_v_dim = True

    def __init__(
        self,
        config,
        layer_number,
        attn_mask_type,
        attention_type,
        softmax_scale=None,
        k_channels=None,
        v_channels=None,
        cp_comm_type=None,
        pg_collection=None,
        **_kwargs,
    ) -> None:
        super().__init__()
        del layer_number, attn_mask_type, attention_type, cp_comm_type
        if pg_collection is None or pg_collection.cp is None:
            raise ValueError("Magi expanded MLA requires an explicit CP process group")
        # Attention.__init__ constructs core_attention once before
        # MultiLatentAttention replaces it with the MLA-specific instance.
        # That first construction does not pass k_channels/v_channels.
        if k_channels is None:
            k_channels = config.qk_head_dim + config.qk_pos_emb_head_dim
        if v_channels is None:
            v_channels = config.v_head_dim
        self.softmax_scale = softmax_scale
        self.k_channels = int(k_channels)
        self.v_channels = int(v_channels)
        # Megatron's THD compatibility shim inspects and temporarily mutates
        # this public field.  Keep the real Magi V dimension separately.
        self.hidden_size_per_attention_head_v = self.v_channels
        self.cp_group = pg_collection.cp
        self._runtime: ExpandedMLARuntime | None = None
        self.last_execution: dict[str, Any] = {}

    def bind_runtime(self, runtime: ExpandedMLARuntime) -> None:
        if runtime.cp_group is not self.cp_group:
            raise ValueError("runtime CP group does not match the backend CP group")
        self._runtime = runtime

    def get_extra_state(self):
        """Match TE core-attention's parameter-free checkpoint surface."""
        return torch.empty(0, dtype=torch.uint8)

    def set_extra_state(self, state) -> None:
        if not isinstance(state, torch.Tensor) or state.dtype != torch.uint8 or state.numel():
            raise ValueError("Magi expanded MLA core attention has no checkpoint state")

    @staticmethod
    def _to_thd(tensor: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if tensor.ndim == 4:
            if tensor.shape[1] != 1:
                raise ValueError("the first Magi MLA adapter supports batch size one per rank")
            return tensor.squeeze(1), True
        if tensor.ndim == 3:
            return tensor, False
        raise ValueError(f"expected THD or SBHD QKV, got shape {tuple(tensor.shape)}")

    def _global_in_physical_order(
        self, local: torch.Tensor, local_to_global: torch.Tensor
    ) -> torch.Tensor:
        world_size = dist.get_world_size(self.cp_group)
        if world_size == 1:
            return local.index_select(0, torch.argsort(local_to_global))

        # The correctness-first adapter currently requires even native CP
        # shards.  torch.distributed.nn.functional.all_gather keeps autograd
        # connected so projection gradients remain comparable to Megatron.
        sizes = [torch.empty(1, dtype=torch.int64, device=local.device) for _ in range(world_size)]
        local_size = torch.tensor([local.shape[0]], dtype=torch.int64, device=local.device)
        dist.all_gather(sizes, local_size, group=self.cp_group)
        lengths = [int(size.item()) for size in sizes]
        if len(set(lengths)) != 1:
            raise NotImplementedError("uneven native CP shards are not supported by T08")

        from torch.distributed.nn.functional import all_gather as autograd_all_gather

        gathered = autograd_all_gather(local, group=self.cp_group)
        gathered_ids = [torch.empty_like(local_to_global) for _ in range(world_size)]
        dist.all_gather(gathered_ids, local_to_global, group=self.cp_group)
        global_tensor = torch.cat(tuple(gathered), dim=0)
        global_ids = torch.cat(gathered_ids, dim=0)
        expected = torch.arange(global_ids.numel(), device=global_ids.device)
        if not torch.equal(torch.sort(global_ids).values, expected):
            raise RuntimeError("native CP mapping does not cover global physical token order")
        return global_tensor.index_select(0, torch.argsort(global_ids))

    def forward(
        self,
        query,
        key,
        value,
        attention_mask=None,
        packed_seq_params=None,
        attn_mask_type=None,
        **_kwargs,
    ):
        del attention_mask, packed_seq_params, attn_mask_type
        if self._runtime is None:
            raise RuntimeError("Magi expanded MLA runtime was not bound for this microbatch")
        if value is None:
            raise NotImplementedError("the first Magi MLA adapter is training-only expanded MLA")

        from magi_attention.api import calc_attn, dispatch, undispatch

        local_q, was_sbhd = self._to_thd(query)
        local_k, key_was_sbhd = self._to_thd(key)
        local_v, value_was_sbhd = self._to_thd(value)
        if key_was_sbhd != was_sbhd or value_was_sbhd != was_sbhd:
            raise ValueError("Q/K/V layouts disagree")
        if local_q.shape[-1] != self.k_channels or local_k.shape[-1] != self.k_channels:
            raise ValueError("Q/K head dimensions do not match the Magi runtime")
        # Megatron pads V to Q/K width for some THD backends.  PR #331
        # natively supports asymmetric QK/V, so remove that compatibility pad.
        if local_v.shape[-1] < self.v_channels:
            raise ValueError("V head dimension is smaller than the configured MLA V dimension")
        local_v = local_v[..., : self.v_channels]

        mapping = self._runtime.local_to_global.to(device=local_q.device, dtype=torch.int64)
        global_q = self._global_in_physical_order(local_q, mapping)
        global_k = self._global_in_physical_order(local_k, mapping)
        global_v = self._global_in_physical_order(local_v, mapping)

        magi_q = dispatch(global_q, self._runtime.key)
        magi_k = dispatch(global_k, self._runtime.key)
        magi_v = dispatch(global_v, self._runtime.key)
        magi_output, _ = calc_attn(
            magi_q,
            magi_k,
            magi_v,
            self._runtime.key,
            softmax_scale=self.softmax_scale,
        )
        global_output = undispatch(magi_output, self._runtime.key)
        native_output = global_output.index_select(0, mapping)
        self.last_execution = {
            "communication_takeover": True,
            "q_shape": tuple(magi_q.shape),
            "k_shape": tuple(magi_k.shape),
            "v_shape": tuple(magi_v.shape),
            "output_shape": tuple(magi_output.shape),
        }
        if was_sbhd:
            return native_output.unsqueeze(1).flatten(start_dim=2)
        return native_output
