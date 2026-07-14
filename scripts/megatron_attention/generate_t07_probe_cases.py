#!/usr/bin/env python3
"""Generate the deterministic T07 native capability-probe cases."""

from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CASE_DIR = ROOT / "exps/megatron_attention/cases/t07_probe"
MANIFEST = ROOT / "exps/megatron_attention/t07_native_probe_manifest.json"


BASE = {
    "schema_version": 1,
    "case_name": "",
    "architecture": "attention",
    "layer_recipe": {"pattern": ["mla"], "repeats": 1},
    "attention_mode": "dense_mla",
    "attention_representation": "expanded_qkv",
    "model": {
        "hidden_size": 512,
        "num_attention_heads": 4,
        "num_query_groups": 4,
        "q_lora_rank": 128,
        "kv_lora_rank": 128,
        "qk_nope_head_dim": 128,
        "qk_pos_emb_head_dim": 64,
        "v_head_dim": 128,
        "vocab_size": 4096,
    },
    "dsa": {"topk": 0},
    "sequence": {
        "layout": "sbhd",
        "logical_lengths": [512],
        "padded_lengths": [512],
    },
    "parallel": {"tp": 1, "cp": 1, "pp": 1, "dp": 1},
    "backend": {
        "name": "megatron_native",
        "layout_policy": "native_zigzag",
        "cp_comm_type": "none",
        "capabilities": [
            "expanded_qkv",
            "dense",
            "bfloat16",
            "fixed_gradient",
            "asymmetric_qk_v_dims",
        ],
    },
    "harness_mode": "attention_stack",
    "dtype": "bfloat16",
    "seed": 1234,
    "warmup": 1,
    "iterations": 3,
    "correctness": {
        "output_atol": 0.001,
        "output_rtol": 0.001,
        "grad_atol": 0.002,
        "grad_rtol": 0.002,
    },
    "loss_policy": "fixed_gradient",
    "claim_scope": "T07 native capability probe; timing is not a headline result",
}


def _case(*, cp: int, mode: str, layout: str = "sbhd", dsa: bool = False) -> dict:
    case = copy.deepcopy(BASE)
    kind = "dsa" if dsa else "dense"
    case["case_name"] = f"t07_probe_{kind}_{layout}_cp{cp}_{mode}"
    case["parallel"]["cp"] = cp
    case["backend"]["cp_comm_type"] = mode
    if cp > 1:
        case["backend"]["capabilities"].append("context_parallel")
    if layout == "thd":
        case["sequence"] = {
            "layout": "thd",
            "logical_lengths": [256, 128, 64, 64],
            "padded_lengths": [256, 128, 64, 64],
        }
        case["backend"]["layout_policy"] = "native_per_document"
        case["backend"]["capabilities"].append("packed_varlen")
    if dsa:
        case["attention_mode"] = "mla_dsa"
        case["attention_representation"] = "absorbed_latent"
        case["dsa"] = {
            "topk": 32,
            "scoring_semantics_id": "megatron_dsa_v1",
            "tie_break": "score_then_global_token_id",
            "score_dtype": "float32",
            "document_local": True,
        }
        capabilities = case["backend"]["capabilities"]
        capabilities[capabilities.index("expanded_qkv")] = "absorbed_latent"
        capabilities[capabilities.index("dense")] = "exact_topk"
    return case


def main() -> None:
    specs = []

    def add(case: dict, *, required: bool, group: str | None = None) -> None:
        path = CASE_DIR / f"{case['case_name']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(case, indent=2) + "\n")
        specs.append(
            {
                "case_id": case["case_name"],
                "config": str(path.relative_to(ROOT)),
                "required": required,
                "selection_group": group,
            }
        )

    add(_case(cp=1, mode="none"), required=True, group="dense_sbhd_cp1")
    for cp in (2, 4):
        for mode in ("p2p", "all_gather", "a2a", "hierarchical"):
            add(
                _case(cp=cp, mode=mode),
                required=False,
                group=f"dense_sbhd_cp{cp}",
            )
    add(_case(cp=1, mode="none", layout="thd"), required=True, group="dense_thd_cp1")
    for cp in (2, 4):
        for mode in ("p2p", "all_gather", "a2a", "a2a+p2p"):
            add(
                _case(cp=cp, mode=mode, layout="thd"),
                required=False,
                group=f"dense_thd_cp{cp}",
            )
    add(_case(cp=1, mode="none", dsa=True), required=True, group="dsa_sbhd_cp1")
    for cp in (2, 4):
        add(
            _case(cp=cp, mode="allgather", dsa=True),
            required=False,
            group=f"dsa_sbhd_cp{cp}",
        )

    MANIFEST.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "repeats": 2,
                "timeout_seconds": 180,
                "purpose": "T07 capability probe; not a headline performance matrix",
                "required_groups": [
                    "dense_sbhd_cp1",
                    "dense_sbhd_cp2",
                    "dense_sbhd_cp4",
                    "dense_thd_cp1",
                    "dense_thd_cp2",
                    "dense_thd_cp4",
                    "dsa_sbhd_cp1",
                    "dsa_sbhd_cp2",
                    "dsa_sbhd_cp4"
                ],
                "cases": specs,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
