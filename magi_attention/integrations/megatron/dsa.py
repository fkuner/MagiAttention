"""Conservative Megatron DSA integration boundary.

This first backend deliberately delegates absorbed MLA, exact top-k, indexer
loss, and sparse attention to the pinned Megatron implementation. It proves
that MagiAttention can replace the DSA core through ModuleSpec without changing
parameters or checkpoint keys. A later backend can take over communication at
this same boundary after parity is established.
"""

from __future__ import annotations

from typing import Optional

import torch
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexerLossLoggingHelper,
    DSAttention,
)


class ConservativeMagiDSAttention(DSAttention):
    """Exact Megatron DSA reference exposed through the Magi backend seam."""

    scoring_semantics_id = "megatron_dsa_v1"
    communication_takeover = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.last_execution: dict[str, object] = {}
        self.last_indexer_loss: torch.Tensor | None = None

    def _tracker_value(self) -> torch.Tensor | None:
        values = DSAIndexerLossLoggingHelper.tracker.get("values")
        if values is None or self.layer_number > values.numel():
            return None
        return values[self.layer_number - 1].detach().clone()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        x: torch.Tensor,
        qr: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attn_mask_type: AttnMaskType = None,
        attention_bias: torch.Tensor = None,
        packed_seq_params: PackedSeqParams = None,
        up_v_weight: Optional[torch.Tensor] = None,
    ):
        before = self._tracker_value()
        output = super().forward(
            query,
            key,
            value,
            attention_mask,
            x,
            qr,
            position_ids=position_ids,
            attn_mask_type=attn_mask_type,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            up_v_weight=up_v_weight,
        )
        after = self._tracker_value()
        if after is not None:
            self.last_indexer_loss = after if before is None else after - before
        cp_group = getattr(self.pg_collection, "cp", None)
        cp_size = cp_group.size() if cp_group is not None else 1
        self.last_execution = {
            "communication_takeover": False,
            "implementation": "megatron_exact_allgather_reference",
            "scoring_semantics_id": self.scoring_semantics_id,
            "topk": self.index_topk,
            "cp_size": cp_size,
            "packed_thd": packed_seq_params is not None
            and packed_seq_params.qkv_format == "thd",
            "absorbed_mla": value is None and up_v_weight is not None,
            "query_shape": tuple(query.shape),
            "key_shape_before_megatron_gather": tuple(key.shape),
            "output_shape": tuple(output.shape),
        }
        return output
