from __future__ import annotations

from pathlib import Path

import pytest
import torch

from exps.megatron_attention.batch import (
    BatchContractError,
    BatchFactory,
    assert_batches_equal,
    assert_legal_kv_selection,
    assert_round_trip,
    causal_document_legality,
    local_loss_oracle,
)
from exps.megatron_attention.config import AttentionBenchmarkCase, load_case

CASES = Path(__file__).parents[2] / "exps" / "megatron_attention" / "cases"
FIXTURES = Path(__file__).parents[2] / "exps" / "megatron_attention" / "fixtures"


def _case(name: str) -> AttentionBenchmarkCase:
    return load_case(CASES / name)


def _case_with_sequence_fixture(case_name: str, fixture_name: str) -> AttentionBenchmarkCase:
    import json

    source = _case(case_name).to_dict()
    for derived in ("selection", "world_size", "config_hash"):
        source.pop(derived)
    source["sequence"] = json.loads((FIXTURES / fixture_name).read_text())
    return AttentionBenchmarkCase.from_dict(source)


@pytest.mark.parametrize(
    "case_name",
    ["smoke_mla_cp1.json", "smoke_packed_cp2.json", "mla_dsa_cp4.json"],
)
def test_factory_is_deterministic_and_clone_equal(case_name: str) -> None:
    case = _case(case_name)
    first = BatchFactory().build(case)
    second = BatchFactory().build(case)
    assert_batches_equal(first, second)
    assert_batches_equal(first, first.clone())


def test_non_divisible_padded_fixture_preserves_logical_and_physical_metadata() -> None:
    case = _case_with_sequence_fixture(
        "smoke_packed_cp2.json", "non_divisible_packed.json"
    )
    batch = BatchFactory().build(case)
    assert batch.cu_seqlens_q.tolist() == [0, 13, 22, 42, 56]
    assert batch.cu_seqlens_q_padded.tolist() == [0, 16, 28, 48, 64]
    assert batch.logical_num_tokens == 56
    assert batch.physical_num_tokens == 64
    assert torch.all(batch.global_token_ids[~batch.valid_token_mask] == -1)
    assert torch.all(batch.loss_mask[~batch.valid_token_mask] == 0)


def test_boundary_heavy_fixture_preserves_every_document_boundary() -> None:
    case = _case_with_sequence_fixture(
        "smoke_packed_cp2.json", "boundary_heavy_packed.json"
    )
    batch = BatchFactory().build(case)
    assert len(batch.cu_seqlens_q) == 17
    assert batch.logical_num_tokens == sum(case.sequence.logical_lengths)
    for document_id, logical_length in enumerate(case.sequence.logical_lengths):
        valid = batch.document_ids == document_id
        assert valid.sum().item() == logical_length
        assert batch.positions_in_document[valid].tolist() == list(range(logical_length))


def test_round_trip_handles_non_contiguous_rank_layout() -> None:
    batch = BatchFactory().build(_case("smoke_packed_cp2.json"))
    ids = torch.arange(batch.physical_num_tokens)
    rank_ids = [ids[::2], ids[1::2]]
    assert_round_trip(batch.tokens, rank_ids)
    assert_round_trip(batch.global_token_ids, rank_ids)


def test_round_trip_rejects_missing_or_duplicate_tokens() -> None:
    batch = BatchFactory().build(_case("smoke_mla_cp1.json"))
    ids = torch.arange(batch.physical_num_tokens)
    with pytest.raises(BatchContractError, match="exactly once"):
        assert_round_trip(batch.tokens, [ids[:-1], ids[-2:-1]])


def test_document_causal_legality_and_cross_document_rejection() -> None:
    batch = BatchFactory().build(_case("smoke_packed_cp2.json"))
    first_doc_query = torch.tensor([10])
    legal_kv = torch.tensor([[0, 5, 10]])
    assert torch.all(causal_document_legality(batch, first_doc_query[:, None], legal_kv))
    second_doc_start = int(batch.cu_seqlens_q_padded[1].item())
    with pytest.raises(BatchContractError, match="illegal cross-document"):
        assert_legal_kv_selection(
            batch, first_doc_query, torch.tensor([[second_doc_start]])
        )


def test_local_loss_reduction_matches_global_valid_token_oracle() -> None:
    batch = BatchFactory().build(_case("smoke_packed_cp2.json"))
    per_token_loss = torch.arange(batch.physical_num_tokens, dtype=torch.float64) / 10
    ids = torch.arange(batch.physical_num_tokens)
    rank_ids = [ids[::2], ids[1::2]]
    numerator, count = local_loss_oracle(per_token_loss, batch.loss_mask, rank_ids)
    expected = (per_token_loss * batch.loss_mask).sum()
    assert torch.equal(numerator, expected)
    assert count.item() == batch.loss_num_tokens
