"""Deterministic global-batch construction and layout correctness oracles."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Iterable, Sequence

import torch
from torch import Tensor

if TYPE_CHECKING:
    from .config import AttentionBenchmarkCase


IGNORE_INDEX = -100


class BatchContractError(ValueError):
    """Raised when token identity, padding, or document legality is violated."""


@dataclass(frozen=True)
class GlobalBatch:
    """Canonical physical token stream shared by all benchmark backends.

    All tensor fields are flat in global physical order. Adapters may reshape
    this stream to SBHD or THD, but may not regenerate its contents.
    """

    tokens: Tensor
    labels: Tensor
    loss_mask: Tensor
    position_ids: Tensor
    valid_token_mask: Tensor
    global_token_ids: Tensor
    physical_token_ids: Tensor
    document_ids: Tensor
    positions_in_document: Tensor
    cu_seqlens_q: Tensor
    cu_seqlens_kv: Tensor
    cu_seqlens_q_padded: Tensor
    cu_seqlens_kv_padded: Tensor
    seed: int
    recipe_hash: str

    @property
    def physical_num_tokens(self) -> int:
        return int(self.tokens.numel())

    @property
    def logical_num_tokens(self) -> int:
        return int(self.valid_token_mask.sum().item())

    @property
    def loss_num_tokens(self) -> int:
        return int(self.loss_mask.sum().item())

    def clone(self) -> "GlobalBatch":
        values = {
            field.name: getattr(self, field.name).clone()
            if isinstance(getattr(self, field.name), Tensor)
            else getattr(self, field.name)
            for field in fields(self)
        }
        return GlobalBatch(**values)


class BatchFactory:
    """Build a deterministic physical batch once for Native and Magi adapters."""

    def __init__(self, *, ignore_index: int = IGNORE_INDEX) -> None:
        self.ignore_index = ignore_index

    def build(self, case: "AttentionBenchmarkCase") -> GlobalBatch:
        case.validate()
        logical_lengths = case.sequence.logical_lengths
        padded_lengths = case.sequence.padded_lengths
        physical_total = sum(padded_lengths)
        logical_total = sum(logical_lengths)

        tokens = torch.zeros(physical_total, dtype=torch.long)
        labels = torch.full((physical_total,), self.ignore_index, dtype=torch.long)
        loss_mask = torch.zeros(physical_total, dtype=torch.float32)
        position_ids = torch.zeros(physical_total, dtype=torch.long)
        valid_mask = torch.zeros(physical_total, dtype=torch.bool)
        global_ids = torch.full((physical_total,), -1, dtype=torch.long)
        physical_ids = torch.arange(physical_total, dtype=torch.long)
        document_ids = torch.full((physical_total,), -1, dtype=torch.long)
        positions = torch.full((physical_total,), -1, dtype=torch.long)

        physical_offset = 0
        logical_offset = 0
        for document_id, (logical, padded) in enumerate(
            zip(logical_lengths, padded_lengths, strict=True)
        ):
            physical_slice = slice(physical_offset, physical_offset + logical)
            local_positions = torch.arange(logical, dtype=torch.long)
            ids = torch.arange(logical_offset, logical_offset + logical, dtype=torch.long)
            document_tokens = self._token_values(
                ids=ids,
                document_id=document_id,
                seed=case.seed,
                vocab_size=case.model.vocab_size,
            )
            tokens[physical_slice] = document_tokens
            valid_mask[physical_slice] = True
            global_ids[physical_slice] = ids
            document_ids[physical_slice] = document_id
            positions[physical_slice] = local_positions
            position_ids[physical_slice] = local_positions
            if logical > 1:
                label_slice = slice(physical_offset, physical_offset + logical - 1)
                labels[label_slice] = document_tokens[1:]
                loss_mask[label_slice] = 1.0
            physical_offset += padded
            logical_offset += logical

        if physical_offset != physical_total or logical_offset != logical_total:
            raise AssertionError("internal batch offset mismatch")

        logical_cu = _cumulative_lengths(logical_lengths)
        padded_cu = _cumulative_lengths(padded_lengths)
        recipe_hash = _recipe_hash(case.seed, logical_lengths, padded_lengths, case.model.vocab_size)
        batch = GlobalBatch(
            tokens=tokens,
            labels=labels,
            loss_mask=loss_mask,
            position_ids=position_ids,
            valid_token_mask=valid_mask,
            global_token_ids=global_ids,
            physical_token_ids=physical_ids,
            document_ids=document_ids,
            positions_in_document=positions,
            cu_seqlens_q=logical_cu,
            cu_seqlens_kv=logical_cu.clone(),
            cu_seqlens_q_padded=padded_cu,
            cu_seqlens_kv_padded=padded_cu.clone(),
            seed=case.seed,
            recipe_hash=recipe_hash,
        )
        validate_global_batch(batch, logical_lengths, padded_lengths, self.ignore_index)
        return batch

    @staticmethod
    def _token_values(
        *, ids: Tensor, document_id: int, seed: int, vocab_size: int
    ) -> Tensor:
        # Avoid RNG state so materialization is bitwise stable across processes.
        usable_vocab = max(vocab_size - 1, 1)
        return ((ids * 104729 + document_id * 8191 + seed) % usable_vocab) + 1


def _cumulative_lengths(lengths: Sequence[int]) -> Tensor:
    values = [0]
    for length in lengths:
        values.append(values[-1] + int(length))
    return torch.tensor(values, dtype=torch.int32)


def _recipe_hash(
    seed: int, logical_lengths: Sequence[int], padded_lengths: Sequence[int], vocab_size: int
) -> str:
    payload = {
        "seed": seed,
        "logical_lengths": list(logical_lengths),
        "padded_lengths": list(padded_lengths),
        "vocab_size": vocab_size,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_global_batch(
    batch: GlobalBatch,
    logical_lengths: Sequence[int],
    padded_lengths: Sequence[int],
    ignore_index: int = IGNORE_INDEX,
) -> None:
    physical_total = sum(padded_lengths)
    logical_total = sum(logical_lengths)
    for field in fields(batch):
        value = getattr(batch, field.name)
        if isinstance(value, Tensor) and not field.name.startswith("cu_seqlens"):
            if value.ndim != 1 or value.numel() != physical_total:
                raise BatchContractError(
                    f"{field.name} must be flat with physical length {physical_total}"
                )
    if batch.logical_num_tokens != logical_total:
        raise BatchContractError("valid-token count does not match logical lengths")
    if not torch.equal(batch.cu_seqlens_q, _cumulative_lengths(logical_lengths)):
        raise BatchContractError("logical cu_seqlens do not match logical lengths")
    if not torch.equal(batch.cu_seqlens_q_padded, _cumulative_lengths(padded_lengths)):
        raise BatchContractError("padded cu_seqlens do not match physical lengths")
    invalid = ~batch.valid_token_mask
    if torch.any(batch.global_token_ids[invalid] != -1):
        raise BatchContractError("padding tokens must use global_token_ids=-1")
    if torch.any(batch.document_ids[invalid] != -1):
        raise BatchContractError("padding tokens must use document_ids=-1")
    if torch.any(batch.positions_in_document[invalid] != -1):
        raise BatchContractError("padding tokens must use positions_in_document=-1")
    if torch.any(batch.labels[invalid] != ignore_index) or torch.any(batch.loss_mask[invalid] != 0):
        raise BatchContractError("padding tokens must be excluded from local loss")
    if torch.any((batch.loss_mask > 0) & ~batch.valid_token_mask):
        raise BatchContractError("loss mask contains invalid tokens")


def assert_batches_equal(left: GlobalBatch, right: GlobalBatch) -> None:
    """Require Native and Magi materializations to share exact token identity."""
    for field in fields(left):
        lhs = getattr(left, field.name)
        rhs = getattr(right, field.name)
        if isinstance(lhs, Tensor):
            if not torch.equal(lhs, rhs):
                raise BatchContractError(f"global batches differ at tensor field {field.name}")
        elif lhs != rhs:
            raise BatchContractError(f"global batches differ at field {field.name}")


def causal_document_legality(
    batch: GlobalBatch, query_physical_ids: Tensor, kv_physical_ids: Tensor
) -> Tensor:
    """Return legality for broadcastable query/KV physical-id tensors."""
    q_valid = batch.valid_token_mask[query_physical_ids]
    kv_valid = batch.valid_token_mask[kv_physical_ids]
    same_document = batch.document_ids[query_physical_ids] == batch.document_ids[kv_physical_ids]
    causal = (
        batch.positions_in_document[kv_physical_ids]
        <= batch.positions_in_document[query_physical_ids]
    )
    return q_valid & kv_valid & same_document & causal


def assert_legal_kv_selection(
    batch: GlobalBatch, query_physical_ids: Tensor, selected_kv_physical_ids: Tensor
) -> None:
    queries = query_physical_ids.reshape(-1, 1).expand_as(selected_kv_physical_ids)
    legal = causal_document_legality(batch, queries, selected_kv_physical_ids)
    if not torch.all(legal):
        first = (~legal).nonzero(as_tuple=False)[0].tolist()
        raise BatchContractError(f"illegal cross-document/causal KV selection at {first}")


def dispatch_by_physical_ids(tensor: Tensor, rank_physical_ids: Sequence[Tensor]) -> list[Tensor]:
    if tensor.ndim == 0:
        raise BatchContractError("cannot dispatch a scalar")
    return [tensor.index_select(0, indices.to(dtype=torch.long)) for indices in rank_physical_ids]


def combine_by_physical_ids(
    local_tensors: Sequence[Tensor], rank_physical_ids: Sequence[Tensor], physical_num_tokens: int
) -> Tensor:
    if len(local_tensors) != len(rank_physical_ids) or not local_tensors:
        raise BatchContractError("local tensors and rank mappings must have the same non-zero size")
    seen = torch.zeros(physical_num_tokens, dtype=torch.int32)
    output_shape = (physical_num_tokens, *local_tensors[0].shape[1:])
    output = torch.empty(output_shape, dtype=local_tensors[0].dtype)
    for local, indices in zip(local_tensors, rank_physical_ids, strict=True):
        indices = indices.to(dtype=torch.long)
        if local.shape[0] != indices.numel():
            raise BatchContractError("local tensor length does not match rank mapping")
        if torch.any(indices < 0) or torch.any(indices >= physical_num_tokens):
            raise BatchContractError("rank mapping contains out-of-range physical ids")
        output.index_copy_(0, indices, local)
        seen.index_add_(0, indices, torch.ones_like(indices, dtype=seen.dtype))
    if torch.any(seen != 1):
        raise BatchContractError("rank mappings must cover each physical token exactly once")
    return output


def assert_round_trip(tensor: Tensor, rank_physical_ids: Sequence[Tensor]) -> None:
    local = dispatch_by_physical_ids(tensor, rank_physical_ids)
    combined = combine_by_physical_ids(local, rank_physical_ids, tensor.shape[0])
    if not torch.equal(tensor, combined):
        raise BatchContractError("dispatch/combine did not restore physical token order")


def local_loss_oracle(
    per_token_loss: Tensor,
    loss_mask: Tensor,
    rank_physical_ids: Iterable[Tensor],
) -> tuple[Tensor, Tensor]:
    """Simulate local loss followed by CP reduction of numerator and token count."""
    numerator = torch.zeros((), dtype=per_token_loss.dtype)
    count = torch.zeros((), dtype=torch.float64)
    for indices in rank_physical_ids:
        local_loss = per_token_loss.index_select(0, indices)
        local_mask = loss_mask.index_select(0, indices).to(dtype=per_token_loss.dtype)
        numerator = numerator + (local_loss * local_mask).sum()
        count = count + local_mask.to(dtype=torch.float64).sum()
    return numerator, count
