"""Focused CPU contracts for the Megatron-native benchmark adapter."""

from types import SimpleNamespace

import torch

from exps.megatron_attention.adapters.native import (
    MegatronNativeAdapter,
    NativeAttentionLifecycle,
)


def test_clone_shared_state_preserves_keys_values_without_aliasing():
    module = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.LayerNorm(3))

    cloned = MegatronNativeAdapter.clone_shared_state(module)
    original = module.state_dict()

    assert cloned.keys() == original.keys()
    for name in original:
        assert torch.equal(cloned[name], original[name])
        assert cloned[name].device.type == "cpu"
        assert cloned[name].data_ptr() != original[name].data_ptr()

    with torch.no_grad():
        next(module.parameters()).add_(1)
    assert not torch.equal(cloned["0.weight"], original["0.weight"])


def test_native_lifecycle_finalize_applies_optional_bias():
    output = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    bias = torch.tensor([1.0, 2.0, 3.0])

    assert NativeAttentionLifecycle.finalize(output, None) is output
    assert torch.equal(
        NativeAttentionLifecycle.finalize(output, bias),
        output + bias,
    )


def test_native_lifecycle_prepare_can_build_non_contiguous_hidden():
    modules = SimpleNamespace(embedding=torch.nn.Embedding(16, 4))
    prepared = SimpleNamespace(tokens=torch.tensor([1, 2, 3], dtype=torch.long))
    case = SimpleNamespace(model=SimpleNamespace(hidden_size=4))

    hidden = NativeAttentionLifecycle.prepare(
        modules,
        prepared,
        case,
        non_contiguous=True,
    )

    assert hidden.shape == (3, 1, 4)
    assert not hidden.is_contiguous()
