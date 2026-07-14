"""Megatron-native MLA/DSA/GDN benchmark adapter.

This module builds real pinned-Megatron modules. It is an attention scaffold,
not a substitute for the later full-GPT functional gate.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any, Mapping

from .base import BaseBenchmarkAdapter


@dataclass
class NativePreparedBatch:
    tokens: Any
    labels: Any
    loss_mask: Any
    position_ids: Any
    local_to_global: Any
    packed_seq_params: Any | None


@dataclass
class NativeOutput:
    hidden_states: Any
    local_to_global: Any


class NativeAttentionLifecycle:
    """Testable lifecycle boundaries without copying Megatron attention forward."""

    @staticmethod
    def prepare(modules, prepared_batch, case, *, non_contiguous: bool = False):
        import torch

        hidden = modules.embedding(prepared_batch.tokens).reshape(
            -1, 1, case.model.hidden_size
        )
        if non_contiguous and hidden.shape[0] > 1:
            hidden = torch.stack((hidden, hidden), dim=-1)[..., 0]
            if hidden.is_contiguous():
                raise RuntimeError("failed to construct non-contiguous hidden-state regression input")
        return hidden

    @staticmethod
    def execute(layer, hidden, attention_mask, prepared_batch):
        return layer(
            hidden,
            attention_mask,
            packed_seq_params=prepared_batch.packed_seq_params,
            position_ids=prepared_batch.position_ids,
        )

    @staticmethod
    def finalize(output, bias):
        return output if bias is None else output + bias


def _make_bundle(embedding, layers):
    import torch.nn as nn

    class NativeModuleBundle(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = embedding
            self.layers = nn.ModuleList(layers)

    return NativeModuleBundle()


class MegatronNativeAdapter(BaseBenchmarkAdapter):
    """Execute the conservative pinned-Megatron reference path."""

    def __init__(self) -> None:
        super().__init__()
        self._owns_dist = False
        self._model_parallel_initialized = False
        self._pg = None
        self._device = None
        self._active_modules = None
        self._qkv_observations: list[dict[str, object]] = []
        self._force_non_contiguous = False
        self._capture_input_gradient = False
        self._last_input_hidden = None

    @property
    def name(self) -> str:
        return "megatron_native"

    def initialize_process_groups(self, case):
        import torch
        import torch.distributed as dist
        from megatron.core import parallel_state
        from megatron.core.process_groups_config import ProcessGroupCollection
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

        if case.parallel.pp != 1:
            raise NotImplementedError("T06 native scaffold supports PP=1 only")
        if not torch.cuda.is_available():
            raise RuntimeError("Megatron native adapter requires CUDA")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        self._device = torch.device("cuda", local_rank)
        if not dist.is_initialized():
            required = ("RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
            missing = [name for name in required if name not in os.environ]
            if missing:
                raise RuntimeError(
                    "native execution must be launched with torchrun; missing "
                    + ", ".join(missing)
                )
            dist.init_process_group(backend="nccl")
            self._owns_dist = True
        if dist.get_world_size() != case.parallel.world_size:
            raise RuntimeError(
                f"torchrun world size {dist.get_world_size()} does not match case "
                f"world size {case.parallel.world_size}"
            )
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=case.parallel.tp,
            pipeline_model_parallel_size=case.parallel.pp,
            context_parallel_size=case.parallel.cp,
            create_gloo_process_groups=True,
        )
        self._model_parallel_initialized = True
        self._pg = ProcessGroupCollection(
            tp=parallel_state.get_tensor_model_parallel_group(),
            cp=parallel_state.get_context_parallel_group(),
        )
        # Megatron modules use both the default torch generator and the model-
        # parallel CUDA RNG tracker during construction. Seed both explicitly
        # so fresh torchrun processes build the same oracle parameters.
        torch.manual_seed(case.seed)
        model_parallel_cuda_manual_seed(case.seed, force_reset_rng=True)
        return self._pg

    def build(self, case, process_groups, shared_state):
        import torch
        import torch.nn as nn
        from megatron.core.extensions.transformer_engine_spec_provider import (
            TESpecProvider,
        )
        from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
            get_dsa_module_spec_for_backend,
            get_gated_delta_net_module_spec,
        )
        from megatron.core.models.gpt.gpt_layer_specs import (
            get_gpt_layer_with_transformer_engine_submodules,
        )
        from megatron.core.ssm.gated_delta_net import GatedDeltaNet
        from megatron.core.transformer.enums import AttnBackend, AttnMaskType
        from megatron.core.transformer.multi_latent_attention import MLASelfAttention
        from megatron.core.transformer.spec_utils import ModuleSpec, build_module
        from megatron.core.transformer.transformer_config import (
            MLATransformerConfig,
            TransformerConfig,
        )
        from megatron.core.utils import init_method_normal, scaled_init_method_normal

        layer_count = len(case.layer_recipe.pattern) * case.layer_recipe.repeats
        common = dict(
            num_layers=layer_count,
            hidden_size=case.model.hidden_size,
            num_attention_heads=case.model.num_attention_heads,
            num_query_groups=case.model.num_query_groups,
            add_bias_linear=False,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            bf16=True,
            params_dtype=torch.bfloat16,
            layernorm_epsilon=1e-6,
            normalization="RMSNorm",
            tensor_model_parallel_size=case.parallel.tp,
            pipeline_model_parallel_size=case.parallel.pp,
            context_parallel_size=case.parallel.cp,
            sequence_parallel=False,
            apply_rope_fusion=False,
            gradient_accumulation_fusion=False,
            use_cpu_initialization=False,
            perform_initialization=True,
            init_method=init_method_normal(0.02),
            output_layer_init_method=scaled_init_method_normal(0.02, layer_count, multiplier=2.0),
            transformer_impl="transformer_engine",
            # DSA validates its only supported CP mode on TransformerConfig
            # before the layer-level ModuleSpec receives cp_comm_type.
            cp_comm_type=(
                case.backend.cp_comm_type
                if case.attention_mode == "mla_dsa" and case.parallel.cp > 1
                else None
            ),
        )
        dsa_enabled = case.attention_mode == "mla_dsa"
        mla_config = MLATransformerConfig(
            **common,
            multi_latent_attention=True,
            q_lora_rank=case.model.q_lora_rank,
            kv_lora_rank=case.model.kv_lora_rank,
            qk_head_dim=case.model.qk_nope_head_dim,
            qk_pos_emb_head_dim=case.model.qk_pos_emb_head_dim,
            v_head_dim=case.model.v_head_dim,
            rope_type="rope",
            rotary_base=10000,
            original_max_position_embeddings=max(case.sequence.padded_lengths),
            kv_channels=case.model.qk_nope_head_dim,
            qk_layernorm=True,
            attention_backend=AttnBackend.unfused,
            experimental_attention_variant="dsa" if dsa_enabled else None,
            dsa_indexer_n_heads=case.model.num_query_groups if dsa_enabled else None,
            dsa_indexer_head_dim=case.model.qk_nope_head_dim if dsa_enabled else None,
            dsa_indexer_topk=case.dsa.topk if dsa_enabled else None,
            dsa_indexer_loss_coeff=0.01 if dsa_enabled else None,
            dsa_indexer_use_sparse_loss=False,
            # The optional Hadamard rotation is applied identically to indexer
            # Q and K, so disabling it preserves QK scores/top-k semantics.
            dsa_indexer_rotate_activation=False,
            dsa_kernel_backend="none",
            calculate_per_token_loss=False,
        )

        backend = TESpecProvider()
        dense_submodules = get_gpt_layer_with_transformer_engine_submodules(
            multi_latent_attention=True
        ).self_attention.submodules
        dense_submodules = self.customize_dense_submodules(dense_submodules, case)
        dense_spec = ModuleSpec(
            module=MLASelfAttention,
            params={"attn_mask_type": AttnMaskType.causal},
            submodules=dense_submodules,
        )
        dsa_spec = (
            get_dsa_module_spec_for_backend(config=mla_config, backend=backend)
            if dsa_enabled
            else None
        )
        if dsa_spec is not None:
            dsa_spec = self.customize_dsa_spec(dsa_spec, case)

        layers = []
        layer_number = 1
        for _ in range(case.layer_recipe.repeats):
            for kind in case.layer_recipe.pattern:
                if kind == "mla":
                    spec = dsa_spec if dsa_spec is not None else dense_spec
                    layer = build_module(
                        spec,
                        config=mla_config,
                        layer_number=layer_number,
                        cp_comm_type=(
                            None
                            if case.backend.cp_comm_type == "none"
                            else case.backend.cp_comm_type
                        ),
                        pg_collection=process_groups,
                    )
                elif kind == "linear_proxy":
                    linear_head_dim = min(128, case.model.hidden_size // 4)
                    gdn_config = TransformerConfig(
                        **common,
                        linear_conv_kernel_dim=4,
                        linear_key_head_dim=linear_head_dim,
                        linear_value_head_dim=linear_head_dim,
                        linear_num_key_heads=case.model.num_attention_heads,
                        linear_num_value_heads=case.model.num_attention_heads,
                        activation_func=torch.nn.functional.silu,
                        experimental_attention_variant="gated_delta_net",
                        linear_attention_freq=[1],
                    )
                    gdn_spec = get_gated_delta_net_module_spec(gdn_config, backend=backend)
                    layer = GatedDeltaNet(
                        gdn_config,
                        submodules=gdn_spec.submodules,
                        layer_number=layer_number,
                        bias=False,
                        conv_bias=False,
                        conv_init=1.0,
                        use_qk_l2norm=True,
                        A_init_range=(1, 16),
                        pg_collection=process_groups,
                    )
                else:
                    raise AssertionError(f"unsupported layer recipe entry: {kind}")
                layers.append(layer)
                if kind == "mla":
                    layer.core_attention.register_forward_pre_hook(
                        self._assert_core_attention_inputs,
                        with_kwargs=True,
                    )
                layer_number += 1

        embedding = nn.Embedding(case.model.vocab_size, case.model.hidden_size)
        bundle = _make_bundle(embedding, layers).to(
            device=self._device, dtype=torch.bfloat16
        )
        bundle.train()
        if shared_state is not None:
            bundle.load_state_dict(shared_state, strict=True)
        self._active_modules = bundle
        return bundle

    def customize_dense_submodules(self, dense_submodules, case):
        """Backend hook preserving the exact Megatron MLA module construction."""
        return dense_submodules

    def customize_dsa_spec(self, dsa_spec, case):
        """Backend hook preserving the exact Megatron absorbed-MLA/DSA spec."""
        return dsa_spec

    @staticmethod
    def clone_shared_state(modules) -> dict[str, Any]:
        """Clone a backend-neutral state snapshot without sharing tensor storage."""
        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in modules.state_dict().items()
        }

    def _assert_core_attention_inputs(self, module, args, kwargs) -> None:
        import torch

        names = ("query", "key", "value")
        observation: dict[str, object] = {"module": type(module).__name__}
        for index, name in enumerate(names):
            tensor = args[index] if index < len(args) else kwargs.get(name)
            if tensor is None:
                observation[name] = None
                continue
            if tensor.dtype != torch.bfloat16:
                raise RuntimeError(f"{name} must be bfloat16, got {tensor.dtype}")
            if tensor.stride(-1) != 1:
                raise RuntimeError(f"{name} final dimension must be contiguous")
            if tensor.ndim not in (3, 4):
                raise RuntimeError(f"{name} must be THD or SBHD, got shape {tuple(tensor.shape)}")
            observation[name] = {
                "shape": list(tensor.shape),
                "stride": list(tensor.stride()),
                "contiguous": tensor.is_contiguous(),
                "dtype": str(tensor.dtype),
            }
        self._qkv_observations.append(observation)

    def dispatch_batch(self, packed_batch, plan, case):
        import torch
        from megatron.core.packed_seq_params import PackedSeqParams
        from megatron.core.utils import get_batch_on_this_cp_rank

        device = self._device
        is_thd = case.sequence.layout == "thd"
        cu = packed_batch.cu_seqlens_q.to(device=device).unsqueeze(0)
        cu_padded = packed_batch.cu_seqlens_q_padded.to(device=device).unsqueeze(0)
        metadata = {
            "cu_seqlens": cu if is_thd else None,
            "cu_seqlens_padded": cu_padded if is_thd else None,
            "max_seqlen": (
                torch.tensor(
                    [max(case.sequence.padded_lengths)], dtype=torch.int32, device=device
                )
                if is_thd
                else None
            ),
            "local_cp_size": None,
            "hybrid_cp_group": None,
        }
        batch = {
            "tokens": packed_batch.tokens.to(device=device).unsqueeze(0),
            "labels": packed_batch.labels.to(device=device).unsqueeze(0),
            "loss_mask": packed_batch.loss_mask.to(device=device).unsqueeze(0),
            "position_ids": packed_batch.position_ids.to(device=device).unsqueeze(0),
            "attention_mask": None,
            **metadata,
        }
        mapping_batch = {
            "tokens": packed_batch.physical_token_ids.to(device=device).unsqueeze(0),
            "labels": None,
            "loss_mask": None,
            "position_ids": None,
            "attention_mask": None,
            **metadata,
        }
        batch = get_batch_on_this_cp_rank(batch, is_hybrid_cp=False, cp_group=self._pg.cp)
        mapping_batch = get_batch_on_this_cp_rank(
            mapping_batch, is_hybrid_cp=False, cp_group=self._pg.cp
        )
        packed_seq_params = None
        if is_thd:
            packed_seq_params = PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=cu.squeeze(0),
                cu_seqlens_kv=cu.squeeze(0),
                cu_seqlens_q_padded=cu_padded.squeeze(0),
                cu_seqlens_kv_padded=cu_padded.squeeze(0),
                max_seqlen_q=max(case.sequence.padded_lengths),
                max_seqlen_kv=max(case.sequence.padded_lengths),
                # This field denotes Megatron Hybrid Context Parallel, not
                # the ordinary context-parallel world size.
                local_cp_size=None,
                cp_group=self._pg.cp,
                total_tokens=int(batch["tokens"].numel()),
            )
        return NativePreparedBatch(
            tokens=batch["tokens"].reshape(-1),
            labels=batch["labels"].reshape(-1),
            loss_mask=batch["loss_mask"].reshape(-1),
            position_ids=batch["position_ids"],
            local_to_global=mapping_batch["tokens"].reshape(-1),
            packed_seq_params=packed_seq_params,
        )

    def forward(self, modules, prepared_batch, case):
        import torch

        self._active_modules = modules
        hidden = NativeAttentionLifecycle.prepare(
            modules,
            prepared_batch,
            case,
            non_contiguous=self._force_non_contiguous,
        )
        if self._capture_input_gradient:
            hidden.retain_grad()
            self._last_input_hidden = hidden
        attention_mask = None
        if case.attention_mode == "mla_dsa" and case.sequence.layout == "sbhd":
            local_len = hidden.shape[0]
            attention_mask = torch.triu(
                torch.full(
                    (1, 1, local_len, local_len),
                    float("-inf"),
                    dtype=torch.float32,
                    device=hidden.device,
                ),
                diagonal=1,
            )
        for layer in modules.layers:
            output, bias = NativeAttentionLifecycle.execute(
                layer,
                hidden,
                attention_mask,
                prepared_batch,
            )
            hidden = NativeAttentionLifecycle.finalize(output, bias)
        return NativeOutput(
            hidden_states=hidden, local_to_global=prepared_batch.local_to_global
        )

    def compute_loss(self, output, prepared_batch, case):
        import torch
        import torch.distributed as dist
        import torch.nn.functional as F

        hidden = output.hidden_states.reshape(-1, case.model.hidden_size)
        if case.loss_policy == "fixed_gradient":
            token_ids = prepared_batch.local_to_global.to(
                device=hidden.device, dtype=torch.int64
            ).reshape(-1, 1)
            feature_ids = torch.arange(
                case.model.hidden_size, device=hidden.device, dtype=torch.int64
            ).reshape(1, -1)
            global_element_ids = token_ids * case.model.hidden_size + feature_ids
            weights = 0.5 + global_element_ids.remainder(1024).float() / 1023.0
            global_numel = sum(case.sequence.padded_lengths) * case.model.hidden_size
            return (hidden.float() * weights).sum() / global_numel
        logits = F.linear(hidden, self._active_modules.embedding.weight)
        per_token = F.cross_entropy(
            logits.float(), prepared_batch.labels, reduction="none", ignore_index=-100
        )
        mask = prepared_batch.loss_mask.to(dtype=per_token.dtype)
        local_numerator = (per_token * mask).sum()
        global_count = mask.sum().detach().clone()
        if case.parallel.cp > 1:
            dist.all_reduce(global_count, group=self._pg.cp)
        if global_count.item() <= 0:
            raise RuntimeError("local LM loss has no valid tokens")
        return local_numerator / global_count

    def backward(self, loss, modules, case) -> None:
        import torch.distributed as dist

        loss.backward()
        if case.parallel.cp > 1:
            for parameter in modules.parameters():
                if parameter.grad is not None:
                    dist.all_reduce(parameter.grad, group=self._pg.cp)

    def zero_grad(self, modules, case) -> None:
        modules.zero_grad(set_to_none=True)

    def canonicalize_for_check(self, output, prepared_batch):
        import torch

        hidden = output.hidden_states.detach()
        if not torch.isfinite(hidden).all():
            raise RuntimeError("native output contains non-finite values")
        if hidden.shape[0] != prepared_batch.local_to_global.numel():
            raise RuntimeError("native output/token mapping length mismatch")
        return self._canonicalize_cp_tensor(hidden, prepared_batch.local_to_global)

    def _canonicalize_cp_tensor(self, local_tensor, local_to_global):
        """Restore global physical-token order for correctness-only artifacts."""
        import torch
        import torch.distributed as dist

        if local_tensor.shape[0] != local_to_global.numel():
            raise RuntimeError("canonical tensor/token mapping length mismatch")
        if not dist.is_initialized() or dist.get_world_size(self._pg.cp) == 1:
            order = torch.argsort(local_to_global)
            return local_tensor.index_select(0, order)

        local_size = torch.tensor(
            [local_tensor.shape[0]], device=local_tensor.device, dtype=torch.int64
        )
        sizes = [torch.empty_like(local_size) for _ in range(dist.get_world_size(self._pg.cp))]
        dist.all_gather(sizes, local_size, group=self._pg.cp)
        lengths = [int(size.item()) for size in sizes]
        max_length = max(lengths)

        padded_tensor = torch.zeros(
            (max_length, *local_tensor.shape[1:]),
            dtype=local_tensor.dtype,
            device=local_tensor.device,
        )
        padded_ids = torch.full(
            (max_length,), -1, dtype=torch.int64, device=local_tensor.device
        )
        padded_tensor[: local_tensor.shape[0]].copy_(local_tensor)
        padded_ids[: local_to_global.numel()].copy_(local_to_global)
        gathered_tensors = [torch.empty_like(padded_tensor) for _ in lengths]
        gathered_ids = [torch.empty_like(padded_ids) for _ in lengths]
        dist.all_gather(gathered_tensors, padded_tensor, group=self._pg.cp)
        dist.all_gather(gathered_ids, padded_ids, group=self._pg.cp)
        global_tensor = torch.cat(
            [tensor[:length] for tensor, length in zip(gathered_tensors, lengths, strict=True)]
        )
        global_ids = torch.cat(
            [ids[:length] for ids, length in zip(gathered_ids, lengths, strict=True)]
        )
        if torch.unique(global_ids).numel() != global_ids.numel():
            raise RuntimeError("CP token mappings overlap while canonicalizing golden tensor")
        expected = torch.arange(global_ids.numel(), device=global_ids.device)
        if not torch.equal(torch.sort(global_ids).values, expected):
            raise RuntimeError("CP token mappings do not cover the global physical stream")
        return global_tensor.index_select(0, torch.argsort(global_ids))

    @staticmethod
    def _tensor_sha256(tensor) -> str:
        import torch

        value = tensor.detach().cpu().contiguous()
        digest = hashlib.sha256()
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def capture_golden(self, modules, prepared_batch, case):
        """Capture layout-invariant correctness artifacts outside the timed path."""
        import torch.distributed as dist

        indexer_loss_helper = None
        if case.attention_mode == "mla_dsa":
            from megatron.core.transformer.experimental_attention_variant.dsa import (
                DSAIndexerLossLoggingHelper,
            )

            indexer_loss_helper = DSAIndexerLossLoggingHelper
            indexer_loss_helper.clean_loss_in_tracker()

        self.zero_grad(modules, case)
        self._capture_input_gradient = True
        self._last_input_hidden = None
        try:
            output = self.forward(modules, prepared_batch, case)
            indexer_loss_values = None
            if indexer_loss_helper is not None:
                values = indexer_loss_helper.tracker.get("values")
                if values is None:
                    raise RuntimeError("DSA golden capture produced no indexer loss")
                indexer_loss_values = values.detach().clone()
            loss = self.compute_loss(output, prepared_batch, case)
            global_loss = loss.detach().clone()
            if case.parallel.cp > 1:
                dist.all_reduce(global_loss, group=self._pg.cp)
            self.backward(loss, modules, case)
            if self._last_input_hidden is None or self._last_input_hidden.grad is None:
                raise RuntimeError("golden capture did not retain the input hidden-state gradient")
            canonical_output = self.canonicalize_for_check(output, prepared_batch)
            canonical_input_grad = self._canonicalize_cp_tensor(
                self._last_input_hidden.grad,
                prepared_batch.local_to_global,
            )
            parameter_grad_hashes = {
                name: self._tensor_sha256(parameter.grad)
                for name, parameter in modules.named_parameters()
                if parameter.grad is not None
            }
            if not parameter_grad_hashes:
                raise RuntimeError("golden capture produced no parameter gradients")
            state_hashes = {
                name: self._tensor_sha256(tensor)
                for name, tensor in modules.state_dict().items()
            }
            result = {
                "loss": float(global_loss.item()),
                "output_sha256": self._tensor_sha256(canonical_output),
                "input_grad_sha256": self._tensor_sha256(canonical_input_grad),
                "state_sha256": state_hashes,
                "parameter_grad_sha256": parameter_grad_hashes,
                "parameter_grad_count": len(parameter_grad_hashes),
                "_tensors": {
                    "output": canonical_output.detach().cpu(),
                    "input_grad": canonical_input_grad.detach().cpu(),
                    "parameter_grads": {
                        name: parameter.grad.detach().cpu()
                        for name, parameter in modules.named_parameters()
                        if parameter.grad is not None
                    },
                },
            }
            if indexer_loss_values is not None:
                result["indexer_loss"] = indexer_loss_values.detach().cpu().tolist()
                result["_tensors"]["indexer_loss"] = indexer_loss_values.detach().cpu()
            return result
        finally:
            if indexer_loss_helper is not None:
                indexer_loss_helper.clean_loss_in_tracker()
            self._capture_input_gradient = False
            self._last_input_hidden = None
            self.zero_grad(modules, case)

    def run_correctness_regressions(self, modules, prepared_batch, case):
        import torch

        self._force_non_contiguous = True
        try:
            output = self.forward(modules, prepared_batch, case)
            loss = output.hidden_states.float().square().mean()
            self.backward(loss, modules, case)
            gradients = [
                parameter.grad
                for parameter in modules.parameters()
                if parameter.grad is not None
            ]
            if not gradients:
                raise RuntimeError("non-contiguous regression produced no parameter gradients")
            if not all(torch.isfinite(gradient).all() for gradient in gradients):
                raise RuntimeError("non-contiguous regression produced NaN/Inf gradients")
            state = self.clone_shared_state(modules)
            if set(state) != set(modules.state_dict()):
                raise RuntimeError("shared state clone changed state-dict keys")
            if any(
                original.numel() > 0
                and original.device.type == "cpu"
                and clone.data_ptr() == original.data_ptr()
                for clone, original in zip(
                    state.values(), modules.state_dict().values(), strict=True
                )
            ):
                raise RuntimeError("shared state clone unexpectedly aliases module storage")
            return {
                "non_contiguous_backward": "pass",
                "finite_parameter_gradients": len(gradients),
                "shared_state_clone": "pass",
                "shared_state_tensor_count": len(state),
            }
        finally:
            self._force_non_contiguous = False
            self.zero_grad(modules, case)

    def correctness_metadata(self, canonical, modules, prepared_batch, case):
        return {
            "shape": list(canonical.shape),
            "dtype": str(canonical.dtype),
            "finite": True,
            "parameter_count": sum(parameter.numel() for parameter in modules.parameters()),
            "state_dict_keys": sorted(modules.state_dict()),
            "local_physical_ids": prepared_batch.local_to_global.detach().cpu().tolist(),
            "qkv_observations": self._qkv_observations,
            "scope": "structural native scaffold; cross-backend parity is a later gate",
            "dsa_indexer_rotate_activation": (
                False if case.attention_mode == "mla_dsa" else None
            ),
        }

    def communication_counters(self) -> Mapping[str, object]:
        return {
            "payload_bytes": None,
            "collective_counts": None,
            "reason": (
                "T06 uses native collectives; profiler-backed counters are introduced later"
            ),
        }

    def synchronize(self) -> None:
        if self._device is not None:
            import torch

            torch.cuda.synchronize(self._device)

    def close(self) -> None:
        try:
            if self._model_parallel_initialized:
                from megatron.core import parallel_state

                parallel_state.destroy_model_parallel()
                self._model_parallel_initialized = False
        finally:
            if self._owns_dist:
                import torch.distributed as dist

                if dist.is_initialized():
                    dist.destroy_process_group()
                self._owns_dist = False
