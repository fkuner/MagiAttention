from __future__ import annotations

import importlib.util
import sys
from dataclasses import fields
from pathlib import Path

import pytest


def _load_contracts_without_magi_runtime():
    """T03 contracts must be testable without importing the CUDA runtime package."""
    path = (
        Path(__file__).parents[2]
        / "magi_attention"
        / "integrations"
        / "megatron"
        / "contracts.py"
    )
    module_name = "_megatron_integration_contracts"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


contracts = _load_contracts_without_magi_runtime()
AbsorbedDSARequest = contracts.AbsorbedDSARequest
AbsorbedMLARequest = contracts.AbsorbedMLARequest
AttentionExecutionResult = contracts.AttentionExecutionResult
ContractError = contracts.ContractError
CPRuntimeContext = contracts.CPRuntimeContext
ExactTopKContract = contracts.ExactTopKContract
ExpandedMLARequest = contracts.ExpandedMLARequest


def _context() -> CPRuntimeContext:
    return CPRuntimeContext(
        backend="test",
        layout_plan_id="plan",
        process_groups=object(),
        dispatch_key=None,
        attention_keys={},
        local_to_global=[0],
        global_to_owner=[0],
        local_position_ids=[0],
        packed_metadata=None,
        local_attention_metadata={},
        microbatch_id=0,
        pipeline_stage=0,
        virtual_pipeline_stage=None,
        loss_policy="local_lm_loss",
    )


def test_request_tags_are_not_user_overridable() -> None:
    expanded = ExpandedMLARequest(1, 2, 3, None, None, _context(), 1.0)
    absorbed = AbsorbedMLARequest(1, 2, None, 3, None, None, _context(), 1.0)
    assert expanded.representation == "expanded_qkv"
    assert absorbed.representation == "absorbed_latent"
    assert not fields(ExpandedMLARequest)[-1].init
    assert not fields(AbsorbedMLARequest)[-1].init


def test_dsa_requires_absorbed_mla_request() -> None:
    topk = ExactTopKContract(8, True, True, "megatron_dsa_v1", "score_then_global_token_id", "float32")
    absorbed = AbsorbedMLARequest(1, 2, None, 3, None, None, _context(), 1.0)
    request = AbsorbedDSARequest(absorbed, 4, 5, None, topk, [0], [0], [0], [True], None)
    assert request.mla.representation == "absorbed_latent"
    assert request.selection == "exact_topk"


def test_topk_contract_rejects_approximate_semantics() -> None:
    with pytest.raises(ContractError, match="document-local"):
        ExactTopKContract(8, True, False, "test", "score_then_global_token_id", "float32")


def test_result_requires_complete_topk_pair() -> None:
    with pytest.raises(ContractError, match="both be present"):
        AttentionExecutionResult(1, [0], False, topk_global_ids=[0])
