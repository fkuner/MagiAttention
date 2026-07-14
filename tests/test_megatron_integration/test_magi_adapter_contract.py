from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from exps.megatron_attention.adapters.magi import (
    MagiDSAReferenceAdapter,
    MagiExpandedMLAAdapter,
)
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
    assert MagiDSAReferenceAdapter().name == "magi_dsa_megatron_reference"


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


def test_dsa_adapter_replaces_only_core_module_class() -> None:
    from megatron.core.transformer.experimental_attention_variant.absorbed_mla import (
        AbsorbedMLASelfAttention,
        AbsorbedMLASelfAttentionSubmodules,
    )
    from megatron.core.transformer.experimental_attention_variant.dsa import (
        DSAIndexer,
        DSAttention,
        DSAttentionSubmodules,
    )
    from megatron.core.transformer.spec_utils import ModuleSpec

    from magi_attention.integrations.megatron.dsa import ConservativeMagiDSAttention

    indexer_spec = ModuleSpec(module=DSAIndexer)
    core_spec = ModuleSpec(
        module=DSAttention,
        params={"sentinel": "core"},
        submodules=DSAttentionSubmodules(indexer=indexer_spec),
    )
    outer = ModuleSpec(
        module=AbsorbedMLASelfAttention,
        params={"sentinel": "outer"},
        submodules=AbsorbedMLASelfAttentionSubmodules(core_attention=core_spec),
    )

    replaced = MagiDSAReferenceAdapter().customize_dsa_spec(outer, SimpleNamespace())

    assert replaced is not outer
    assert replaced.module is outer.module
    assert replaced.params == outer.params
    assert replaced.submodules.core_attention.module is ConservativeMagiDSAttention
    assert replaced.submodules.core_attention.params == core_spec.params
    assert replaced.submodules.core_attention.submodules.indexer is indexer_spec
    assert outer.submodules.core_attention.module is DSAttention


def test_conservative_dsa_backend_is_explicitly_not_communication_takeover() -> None:
    from megatron.core.transformer.experimental_attention_variant.dsa import DSAttention

    from magi_attention.integrations.megatron.dsa import ConservativeMagiDSAttention

    assert issubclass(ConservativeMagiDSAttention, DSAttention)
    assert ConservativeMagiDSAttention.scoring_semantics_id == "megatron_dsa_v1"
    assert ConservativeMagiDSAttention.communication_takeover is False
