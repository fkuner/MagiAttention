"""Backend-neutral Megatron/Magi integration contracts.

Only structural Python types live here. Runtime adapters may place torch tensors,
process groups, and Magi keys in the opaque fields without making this module
import those optional dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Protocol, TypeAlias, runtime_checkable


Tensor: TypeAlias = Any


class ContractError(ValueError):
    """Raised when a typed execution contract is internally inconsistent."""


@dataclass(frozen=True)
class TokenSpan:
    global_start: int
    physical_length: int
    valid_length: int
    document_id: int
    position_start: int

    def __post_init__(self) -> None:
        if self.global_start < 0 or self.physical_length <= 0:
            raise ContractError("TokenSpan requires non-negative start and positive physical length")
        if not 0 <= self.valid_length <= self.physical_length:
            raise ContractError("TokenSpan.valid_length must be within physical_length")


@dataclass(frozen=True)
class RankPartition:
    spans_in_local_order: tuple[TokenSpan, ...]


@dataclass(frozen=True)
class CPLayoutPlan:
    schema_version: int
    plan_id: str
    batch_recipe_hash: str
    cp_size: int
    rank_partitions: tuple[RankPartition, ...]
    attention_pattern_ids: tuple[str, ...]
    solver_name: str
    solver_version: str
    solver_config: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.schema_version <= 0 or self.cp_size <= 0:
            raise ContractError("CPLayoutPlan schema_version and cp_size must be positive")
        if len(self.rank_partitions) != self.cp_size:
            raise ContractError("CPLayoutPlan must contain one RankPartition per CP rank")
        if not self.plan_id or not self.batch_recipe_hash:
            raise ContractError("CPLayoutPlan requires stable plan and batch recipe identifiers")


@dataclass(frozen=True)
class PackedMetadata:
    cu_seqlens: Tensor
    cu_seqlens_padded: Tensor
    valid_token_mask: Tensor
    document_ids: Tensor
    positions_in_document: Tensor


@dataclass
class CPRuntimeContext:
    backend: str
    layout_plan_id: str
    process_groups: object
    dispatch_key: object | None
    attention_keys: Mapping[str, object]
    local_to_global: Tensor
    global_to_owner: Tensor
    local_position_ids: Tensor
    packed_metadata: PackedMetadata | None
    local_attention_metadata: object
    microbatch_id: int
    pipeline_stage: int
    virtual_pipeline_stage: int | None
    loss_policy: str


@dataclass
class PreparedBatch:
    local_tokens: Tensor
    local_labels: Tensor | None
    local_loss_mask: Tensor | None
    local_position_ids: Tensor
    packed_seq_params: object | None
    runtime_context: CPRuntimeContext | None
    local_to_global: Tensor


@runtime_checkable
class MagiIntegration(Protocol):
    def capabilities(self) -> object: ...

    def validate_case(self, case: object) -> None: ...

    def bind(
        self,
        plan: CPLayoutPlan,
        packed_metadata: PackedMetadata | None,
        local_attention_metadata: object,
        process_groups: object,
        microbatch_identity: object,
    ) -> CPRuntimeContext: ...


@runtime_checkable
class CPLayoutProvider(Protocol):
    def plan(self, global_batch_metadata: object, case: object) -> CPLayoutPlan: ...

    def bind(
        self,
        plan: CPLayoutPlan,
        local_attention_metadata: object,
        process_groups: object,
        microbatch_identity: object,
    ) -> CPRuntimeContext: ...

    def dispatch(self, tensor: Tensor, context: CPRuntimeContext, *, pad_value: object) -> Tensor: ...

    def combine(self, tensor: Tensor, context: CPRuntimeContext) -> Tensor: ...


@dataclass(frozen=True)
class ExpandedMLARequest:
    query: Tensor
    key: Tensor
    value: Tensor
    attention_mask: object | None
    packed_metadata: PackedMetadata | None
    runtime_context: CPRuntimeContext
    softmax_scale: float
    representation: Literal["expanded_qkv"] = field(init=False, default="expanded_qkv")


@dataclass(frozen=True)
class AbsorbedMLARequest:
    absorbed_query: Tensor
    latent_kv: Tensor
    positional_key: Tensor | None
    up_v_weight: Tensor
    attention_mask: object | None
    packed_metadata: PackedMetadata | None
    runtime_context: CPRuntimeContext
    softmax_scale: float
    representation: Literal["absorbed_latent"] = field(init=False, default="absorbed_latent")


@dataclass(frozen=True)
class ExactTopKContract:
    topk: int
    causal: bool
    document_local: bool
    scoring_semantics_id: str
    tie_break: Literal["score_then_global_token_id"]
    score_dtype: str

    def __post_init__(self) -> None:
        if self.topk <= 0:
            raise ContractError("ExactTopKContract.topk must be positive")
        if not self.causal or not self.document_local:
            raise ContractError("the first exact-top-k contract requires causal document-local selection")
        if self.tie_break != "score_then_global_token_id":
            raise ContractError("exact top-k requires deterministic global-token tie breaking")


@dataclass(frozen=True)
class AbsorbedDSARequest:
    mla: AbsorbedMLARequest
    indexer_query: Tensor
    indexer_key: Tensor
    indexer_weights: Tensor | None
    topk_contract: ExactTopKContract
    query_global_ids: Tensor
    kv_global_ids: Tensor
    document_ids: Tensor
    valid_token_mask: Tensor
    index_share_input: object | None
    selection: Literal["exact_topk"] = field(init=False, default="exact_topk")


AttentionExecutionRequest: TypeAlias = ExpandedMLARequest | AbsorbedMLARequest | AbsorbedDSARequest


@dataclass
class AttentionExecutionResult:
    local_output: Tensor
    output_global_ids: Tensor
    v_up_applied: bool
    topk_global_ids: Tensor | None = None
    topk_scores: Tensor | None = None
    indexer_loss: Tensor | None = None
    communication_takeover: bool = False

    def __post_init__(self) -> None:
        if (self.topk_global_ids is None) != (self.topk_scores is None):
            raise ContractError("top-k ids and scores must either both be present or both be absent")


@runtime_checkable
class AttentionExecutionBackend(Protocol):
    def execute(self, request: AttentionExecutionRequest) -> AttentionExecutionResult: ...


@runtime_checkable
class BenchmarkBackend(Protocol):
    def initialize_process_groups(self, case: object) -> object: ...

    def build(self, case: object, process_groups: object, shared_state: object) -> object: ...

    def prepare_batch(self, global_batch: object, case: object) -> PreparedBatch: ...

    def forward(self, modules: object, prepared_batch: PreparedBatch, case: object) -> object: ...

    def compute_loss(self, output: object, prepared_batch: PreparedBatch, case: object) -> Tensor: ...

    def backward(self, loss: Tensor, modules: object, case: object) -> None: ...

    def zero_grad(self, modules: object, case: object) -> None: ...

    def canonicalize_for_check(self, output: object, prepared_batch: PreparedBatch) -> Tensor: ...

    def communication_counters(self) -> Mapping[str, int | float | None]: ...
