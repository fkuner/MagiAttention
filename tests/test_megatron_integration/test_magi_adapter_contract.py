from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from exps.megatron_attention.adapters.magi import MagiExpandedMLAAdapter
from magi_attention.integrations.megatron.expanded_mla import (
    MagiExpandedMLACoreAttention,
)


def _backend() -> MagiExpandedMLACoreAttention:
    config = SimpleNamespace(qk_head_dim=128, qk_pos_emb_head_dim=64, v_head_dim=128)
    group = object()
    return MagiExpandedMLACoreAttention(
        config=config,
        layer_number=1,
        attn_mask_type="causal",
        attention_type="self",
        pg_collection=SimpleNamespace(cp=group),
    )


def test_adapter_name_is_explicit() -> None:
    assert MagiExpandedMLAAdapter().name == "magi_expanded_mla"


def test_core_backend_infers_mla_asymmetric_dims() -> None:
    backend = _backend()
    assert backend.k_channels == 192
    assert backend.v_channels == 128
    assert backend.supports_mla_asymmetric_v_dim is True
    assert list(backend.parameters()) == []


def test_core_backend_checkpoint_surface_matches_te_empty_extra_state() -> None:
    backend = _backend()
    state = backend.get_extra_state()
    assert state.dtype == torch.uint8
    assert state.numel() == 0
    backend.set_extra_state(state)
    with pytest.raises(ValueError):
        backend.set_extra_state(torch.ones(1, dtype=torch.uint8))


def test_to_thd_accepts_sbhd_and_thd() -> None:
    sbhd = torch.zeros(8, 1, 4, 192)
    thd, was_sbhd = MagiExpandedMLACoreAttention._to_thd(sbhd)
    assert thd.shape == (8, 4, 192)
    assert was_sbhd is True

    original_thd = torch.zeros(8, 4, 192)
    thd, was_sbhd = MagiExpandedMLACoreAttention._to_thd(original_thd)
    assert thd is original_thd
    assert was_sbhd is False


def test_to_thd_rejects_multi_batch_sbhd() -> None:
    with pytest.raises(ValueError, match="batch size one"):
        MagiExpandedMLACoreAttention._to_thd(torch.zeros(8, 2, 4, 192))
