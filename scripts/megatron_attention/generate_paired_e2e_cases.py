#!/usr/bin/env python3
"""Generate the correctness-gated Native/Magi paired E2E matrix."""

from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CASE_DIR = ROOT / "exps/megatron_attention/cases/e2e_paired"
MANIFEST = ROOT / "exps/megatron_attention/e2e_paired_manifest.json"

SCENARIOS = (
    ("dense_sbhd_cp1", "t08_native_mla_cp1.json", "t08_magi_mla_cp1.json"),
    ("dense_sbhd_cp2", "t08_native_mla_cp2.json", "t08_magi_mla_cp2.json"),
    ("dense_sbhd_cp4", "t08_native_mla_cp4.json", "t08_magi_mla_cp4.json"),
    ("dense_thd_cp1", "t08_native_mla_thd_cp2.json", "t08_magi_mla_thd_cp2.json"),
    ("dense_thd_cp2", "t08_native_mla_thd_cp2.json", "t08_magi_mla_thd_cp2.json"),
    ("dsa_sbhd_cp1", "t09_native_dsa_cp1.json", "t09_magi_dsa_cp1.json"),
    ("dsa_sbhd_cp2", "t09_native_dsa_cp2.json", "t09_magi_dsa_cp2.json"),
    ("dsa_sbhd_cp4", "t09_native_dsa_cp4.json", "t09_magi_dsa_cp4.json"),
    ("dsa_thd_cp2", "t09_native_dsa_thd_cp2.json", "t09_magi_dsa_thd_cp2.json"),
)


def _load(name: str) -> dict:
    path = ROOT / "exps/megatron_attention/cases" / name
    return json.loads(path.read_text())


def _with_harness(case: dict, *, pair_id: str, adapter: str, harness: str) -> dict:
    result = copy.deepcopy(case)
    # The older T08 smoke cases used weak scaling.  A paired golden matrix
    # needs the same global batch at CP1/2/4 so CP1 can serve as an oracle.
    if pair_id.startswith("dense_sbhd_"):
        result["sequence"] = {
            "layout": "sbhd",
            "logical_lengths": [1024],
            "padded_lengths": [1024],
        }
    if pair_id.startswith("dense_thd_cp1_"):
        result["parallel"]["cp"] = 1
    result["case_name"] = f"e2e_{pair_id}_{adapter}"
    result["harness_mode"] = harness
    result["warmup"] = 2
    result["iterations"] = 5
    capabilities = result["backend"]["capabilities"]
    if harness == "attention_stack":
        old, new = "local_lm_loss", "fixed_gradient"
        result["loss_policy"] = new
        scope = "attention-layer E2E"
    else:
        old, new = "fixed_gradient", "local_lm_loss"
        result["loss_policy"] = new
        scope = "one-layer-training E2E"
    if old in capabilities:
        capabilities[capabilities.index(old)] = new
    elif new not in capabilities:
        capabilities.append(new)
    result["claim_scope"] = (
        f"Paired {scope} from batch generation through dispatch, forward, loss, "
        f"backward, and gradient zeroing; adapter={adapter}"
    )
    return result


def main() -> None:
    pairs = []
    CASE_DIR.mkdir(parents=True, exist_ok=True)
    for scenario, native_name, magi_name in SCENARIOS:
        for harness in ("attention_stack", "attention_to_loss"):
            pair_id = f"{scenario}_{harness}"
            paths = {}
            for adapter, source in (("native", native_name), ("magi", magi_name)):
                case = _with_harness(
                    _load(source), pair_id=pair_id, adapter=adapter, harness=harness
                )
                path = CASE_DIR / f"{case['case_name']}.json"
                path.write_text(json.dumps(case, indent=2) + "\n")
                paths[adapter] = str(path.relative_to(ROOT))
            pairs.append(
                {
                    "pair_id": pair_id,
                    "native_config": paths["native"],
                    "magi_config": paths["magi"],
                }
            )
    MANIFEST.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "repeats": 2,
                "timeout_seconds": 300,
                "purpose": (
                    "Correctness-gated attention-layer and one-layer-training E2E "
                    "Native/Magi comparison beginning at batch generation; dense "
                    "SBHD uses a fixed global length across CP1/2/4"
                ),
                "pairs": pairs,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
