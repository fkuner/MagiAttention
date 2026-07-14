from __future__ import annotations

import json
from pathlib import Path

import pytest

from exps.megatron_attention.config import ConfigError, load_case

CASES = Path(__file__).parents[2] / "exps" / "megatron_attention" / "cases"


@pytest.mark.parametrize("path", sorted(CASES.glob("*.json")), ids=lambda path: path.stem)
def test_all_cases_are_valid(path: Path) -> None:
    case = load_case(path)
    assert case.config_hash() == case.to_dict()["config_hash"]
    assert case.parallel.world_size <= 4


def test_dsa_requires_absorbed_representation(tmp_path: Path) -> None:
    payload = json.loads((CASES / "mla_dsa_cp4.json").read_text())
    payload["attention_representation"] = "expanded_qkv"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ConfigError, match="requires attention_representation=absorbed_latent"):
        load_case(path)


def test_hybrid_pattern_is_exact(tmp_path: Path) -> None:
    payload = json.loads((CASES / "hybrid_3to1_cp1.json").read_text())
    payload["layer_recipe"]["pattern"] = ["linear_proxy", "mla"]
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ConfigError, match="requires exactly"):
        load_case(path)


def test_missing_backend_capability_fails_before_runtime(tmp_path: Path) -> None:
    payload = json.loads((CASES / "smoke_mla_cp4.json").read_text())
    payload["backend"]["capabilities"].remove("asymmetric_qk_v_dims")
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ConfigError, match="asymmetric_qk_v_dims"):
        load_case(path)


def test_unknown_fields_are_rejected(tmp_path: Path) -> None:
    payload = json.loads((CASES / "smoke_mla_cp1.json").read_text())
    payload["magi_magic"] = True
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ConfigError, match="unknown fields"):
        load_case(path)
