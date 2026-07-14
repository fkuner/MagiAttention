#!/usr/bin/env python3
"""Recheck preserved Native/Magi golden artifacts with current parity gates."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(f"_{path.stem}_for_recheck", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


paired_matrix = _load_script("run_paired_e2e_matrix.py")


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", action="append", default=[])
    args = parser.parse_args()

    source = _load(args.summary)
    result: dict[str, Any] = {
        "schema_version": 1,
        "source_summary": str(args.summary),
        "purpose": "Offline parity recheck of preserved golden artifacts",
        "pairs": [],
    }
    selected = set(args.only)
    for source_pair in source["pairs"]:
        if selected and source_pair["pair_id"] not in selected:
            continue
        native_config = _load(Path(source_pair["native_config"]))
        correctness = native_config["correctness"]
        pair = {
            "pair_id": source_pair["pair_id"],
            "repeat_count": source_pair["repeat_count"],
            "repeats": [],
        }
        result["pairs"].append(pair)
        for source_repeat in source_pair["repeats"]:
            repeat = int(source_repeat["repeat"])
            try:
                parity = paired_matrix._compare_pair(
                    source_repeat["native"],
                    source_repeat["magi"],
                    output=args.output_dir
                    / f"{source_pair['pair_id']}.repeat{repeat}.strict.parity.json",
                    output_atol=correctness["output_atol"],
                    output_rtol=correctness["output_rtol"],
                    grad_atol=correctness["grad_atol"],
                    grad_rtol=correctness["grad_rtol"],
                )
            except Exception as error:  # preserve evidence from every other repeat
                parity = {
                    "status": "error",
                    "allclose": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            pair["repeats"].append({"repeat": repeat, "parity": parity})
            paired_matrix.native_matrix._atomic_json(result, args.output)
        pair["status"] = (
            "pass"
            if len(pair["repeats"]) == pair["repeat_count"]
            and all(item["parity"].get("allclose") for item in pair["repeats"])
            else "fail"
        )
    result["status"] = (
        "pass" if all(pair["status"] == "pass" for pair in result["pairs"]) else "fail"
    )
    paired_matrix.native_matrix._atomic_json(result, args.output)
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
