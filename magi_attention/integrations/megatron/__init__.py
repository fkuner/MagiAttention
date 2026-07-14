"""Stable contracts for integrating MagiAttention with Megatron Core."""

from .contracts import (
    AbsorbedDSARequest,
    AbsorbedMLARequest,
    AttentionExecutionRequest,
    AttentionExecutionResult,
    ExpandedMLARequest,
)

__all__ = [
    "AbsorbedDSARequest",
    "AbsorbedMLARequest",
    "AttentionExecutionRequest",
    "AttentionExecutionResult",
    "ExpandedMLARequest",
]
