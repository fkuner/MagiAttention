"""Pure-Python configuration schema for the Megatron attention benchmark.

This module intentionally does not import torch, Megatron, or MagiAttention.
All static capability failures must be reported before distributed initialization.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

SCHEMA_VERSION = 1

Architecture = Literal["attention", "hybrid_3to1"]
AttentionMode = Literal["dense_mla", "mla_dsa"]
AttentionRepresentation = Literal["expanded_qkv", "absorbed_latent"]
BackendName = Literal["megatron_native", "magi"]
HarnessMode = Literal["attention_stack", "attention_to_loss"]
LossPolicy = Literal["fixed_gradient", "local_lm_loss"]
SequenceLayout = Literal["sbhd", "thd"]


class ConfigError(ValueError):
    """Raised when a case is invalid before distributed initialization."""


def _require_keys(data: Mapping[str, Any], keys: Sequence[str], where: str) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise ConfigError(f"{where} is missing required fields: {', '.join(missing)}")


def _reject_unknown(data: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigError(f"{where} has unknown fields: {', '.join(unknown)}")


@dataclass(frozen=True)
class LayerRecipe:
    pattern: tuple[str, ...] = ("linear_proxy", "linear_proxy", "linear_proxy", "mla")
    repeats: int = 1

    def validate(self, architecture: Architecture) -> None:
        if self.repeats < 1:
            raise ConfigError("layer_recipe.repeats must be positive")
        if architecture == "hybrid_3to1":
            expected = ("linear_proxy", "linear_proxy", "linear_proxy", "mla")
            if self.pattern != expected:
                raise ConfigError(
                    "hybrid_3to1 requires exactly "
                    "['linear_proxy', 'linear_proxy', 'linear_proxy', 'mla']"
                )
        elif self.pattern != ("mla",):
            raise ConfigError("attention architecture requires layer_recipe.pattern=['mla']")


@dataclass(frozen=True)
class ModelDims:
    hidden_size: int
    num_attention_heads: int
    num_query_groups: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_pos_emb_head_dim: int
    v_head_dim: int
    vocab_size: int

    @property
    def execution_qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_pos_emb_head_dim

    def validate(self, tp: int) -> None:
        for name, value in asdict(self).items():
            if value <= 0:
                raise ConfigError(f"model.{name} must be positive")
        if self.hidden_size % tp:
            raise ConfigError("model.hidden_size must be divisible by TP")
        if self.num_attention_heads % tp:
            raise ConfigError("model.num_attention_heads must be divisible by TP")
        if self.num_query_groups % tp:
            raise ConfigError("model.num_query_groups must be divisible by TP in the first version")
        if self.num_attention_heads % self.num_query_groups:
            raise ConfigError("model.num_attention_heads must be divisible by num_query_groups")


@dataclass(frozen=True)
class DSAConfig:
    topk: int = 0
    scoring_semantics_id: str = "megatron_dsa_v1"
    tie_break: str = "score_then_global_token_id"
    score_dtype: str = "float32"
    document_local: bool = True

    def validate(self, enabled: bool) -> None:
        if enabled and self.topk <= 0:
            raise ConfigError("dsa.topk must be positive for mla_dsa")
        if not enabled and self.topk != 0:
            raise ConfigError("dsa.topk must be 0 for dense_mla")
        if self.tie_break != "score_then_global_token_id":
            raise ConfigError("only deterministic score_then_global_token_id tie breaking is supported")
        if enabled and not self.document_local:
            raise ConfigError("the first DSA version requires document-local selection")


@dataclass(frozen=True)
class SequenceRecipe:
    layout: SequenceLayout
    logical_lengths: tuple[int, ...]
    padded_lengths: tuple[int, ...]

    @property
    def packed(self) -> bool:
        return self.layout == "thd"

    def validate(self) -> None:
        if not self.logical_lengths:
            raise ConfigError("sequence.logical_lengths must not be empty")
        if len(self.logical_lengths) != len(self.padded_lengths):
            raise ConfigError("logical_lengths and padded_lengths must have the same size")
        for index, (logical, padded) in enumerate(
            zip(self.logical_lengths, self.padded_lengths, strict=True)
        ):
            if logical <= 0 or padded <= 0:
                raise ConfigError(f"sequence lengths at index {index} must be positive")
            if logical > padded:
                raise ConfigError(f"logical length exceeds padded length at index {index}")
        if self.layout == "sbhd" and len(self.logical_lengths) != 1:
            raise ConfigError("sbhd requires exactly one sequence length")


@dataclass(frozen=True)
class ParallelConfig:
    tp: int
    cp: int
    pp: int
    dp: int

    @property
    def world_size(self) -> int:
        return self.tp * self.cp * self.pp * self.dp

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if value <= 0:
                raise ConfigError(f"parallel.{name} must be positive")
        if self.world_size > 4:
            raise ConfigError(
                f"TP*CP*PP*DP={self.world_size} exceeds the frozen 4-GPU scaffold environment"
            )


@dataclass(frozen=True)
class BackendConfig:
    name: BackendName
    layout_policy: str
    cp_comm_type: str
    capabilities: tuple[str, ...] = ()


@dataclass(frozen=True)
class CorrectnessEnvelope:
    output_atol: float = 1e-3
    output_rtol: float = 1e-3
    grad_atol: float = 2e-3
    grad_rtol: float = 2e-3

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if value < 0:
                raise ConfigError(f"correctness.{name} must be non-negative")


@dataclass(frozen=True)
class AttentionBenchmarkCase:
    schema_version: int
    case_name: str
    architecture: Architecture
    layer_recipe: LayerRecipe
    attention_mode: AttentionMode
    attention_representation: AttentionRepresentation
    model: ModelDims
    dsa: DSAConfig
    sequence: SequenceRecipe
    parallel: ParallelConfig
    backend: BackendConfig
    harness_mode: HarnessMode
    dtype: str
    seed: int
    warmup: int
    iterations: int
    correctness: CorrectnessEnvelope
    loss_policy: LossPolicy
    claim_scope: str
    expected_capabilities: tuple[str, ...] = field(default_factory=tuple)

    @property
    def selection(self) -> Literal["dense", "exact_topk"]:
        return "exact_topk" if self.attention_mode == "mla_dsa" else "dense"

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ConfigError(
                f"unsupported schema_version={self.schema_version}; expected {SCHEMA_VERSION}"
            )
        if not self.case_name or any(char.isspace() for char in self.case_name):
            raise ConfigError("case_name must be a non-empty identifier without whitespace")
        self.parallel.validate()
        self.layer_recipe.validate(self.architecture)
        self.model.validate(self.parallel.tp)
        self.sequence.validate()
        self.correctness.validate()
        dsa_enabled = self.attention_mode == "mla_dsa"
        self.dsa.validate(dsa_enabled)
        if dsa_enabled and self.attention_representation != "absorbed_latent":
            raise ConfigError("headline mla_dsa requires attention_representation=absorbed_latent")
        if self.dtype != "bfloat16":
            raise ConfigError("the first benchmark version supports dtype=bfloat16 only")
        if self.warmup < 0 or self.iterations <= 0:
            raise ConfigError("warmup must be non-negative and iterations must be positive")
        if self.harness_mode == "attention_to_loss" and self.loss_policy != "local_lm_loss":
            raise ConfigError("attention_to_loss requires loss_policy=local_lm_loss")
        if self.harness_mode == "attention_stack" and self.loss_policy != "fixed_gradient":
            raise ConfigError("attention_stack requires loss_policy=fixed_gradient")
        if not self.claim_scope:
            raise ConfigError("claim_scope must describe what this case may prove")
        self._validate_backend_capabilities()

    def _validate_backend_capabilities(self) -> None:
        caps = set(self.backend.capabilities)
        required = {
            self.attention_representation,
            self.selection,
            self.dtype,
            self.loss_policy,
        }
        if self.sequence.packed:
            required.add("packed_varlen")
        if self.model.execution_qk_head_dim != self.model.v_head_dim:
            required.add("asymmetric_qk_v_dims")
        if self.parallel.cp > 1:
            required.add("context_parallel")
        missing = sorted(required - caps)
        if missing:
            raise ConfigError(
                f"backend {self.backend.name!r} lacks required capabilities: {', '.join(missing)}"
            )
        expected_missing = sorted(set(self.expected_capabilities) - caps)
        if expected_missing:
            raise ConfigError(
                "expected_capabilities are not declared by the backend: "
                + ", ".join(expected_missing)
            )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["selection"] = self.selection
        payload["world_size"] = self.parallel.world_size
        payload["config_hash"] = self.config_hash()
        return payload

    def config_hash(self) -> str:
        payload = asdict(self)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AttentionBenchmarkCase":
        allowed = {
            "schema_version",
            "case_name",
            "architecture",
            "layer_recipe",
            "attention_mode",
            "attention_representation",
            "model",
            "dsa",
            "sequence",
            "parallel",
            "backend",
            "harness_mode",
            "dtype",
            "seed",
            "warmup",
            "iterations",
            "correctness",
            "loss_policy",
            "claim_scope",
            "expected_capabilities",
        }
        _reject_unknown(data, allowed, "case")
        _require_keys(data, sorted(allowed - {"expected_capabilities"}), "case")
        layer_data = dict(data["layer_recipe"])
        sequence_data = dict(data["sequence"])
        backend_data = dict(data["backend"])
        case = cls(
            schema_version=int(data["schema_version"]),
            case_name=str(data["case_name"]),
            architecture=data["architecture"],
            layer_recipe=LayerRecipe(
                pattern=tuple(layer_data["pattern"]), repeats=int(layer_data.get("repeats", 1))
            ),
            attention_mode=data["attention_mode"],
            attention_representation=data["attention_representation"],
            model=ModelDims(**data["model"]),
            dsa=DSAConfig(**data["dsa"]),
            sequence=SequenceRecipe(
                layout=sequence_data["layout"],
                logical_lengths=tuple(sequence_data["logical_lengths"]),
                padded_lengths=tuple(sequence_data["padded_lengths"]),
            ),
            parallel=ParallelConfig(**data["parallel"]),
            backend=BackendConfig(
                name=backend_data["name"],
                layout_policy=backend_data["layout_policy"],
                cp_comm_type=backend_data["cp_comm_type"],
                capabilities=tuple(backend_data["capabilities"]),
            ),
            harness_mode=data["harness_mode"],
            dtype=str(data["dtype"]),
            seed=int(data["seed"]),
            warmup=int(data["warmup"]),
            iterations=int(data["iterations"]),
            correctness=CorrectnessEnvelope(**data["correctness"]),
            loss_policy=data["loss_policy"],
            claim_scope=str(data["claim_scope"]),
            expected_capabilities=tuple(data.get("expected_capabilities", ())),
        )
        case.validate()
        return case


def load_case(path: str | Path) -> AttentionBenchmarkCase:
    case_path = Path(path)
    try:
        payload = json.loads(case_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"failed to load {case_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError(f"case root must be a JSON object: {case_path}")
    return AttentionBenchmarkCase.from_dict(payload)
