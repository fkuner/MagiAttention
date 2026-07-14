#!/usr/bin/env python3
"""Run fair Native/Magi E2E pairs with alternating launch order."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def _load_native_matrix_module():
    path = Path(__file__).with_name("run_native_matrix.py")
    spec = importlib.util.spec_from_file_location("_native_matrix_for_pairs", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


native_matrix = _load_native_matrix_module()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _run_case(
    *,
    config: Path,
    adapter: str,
    output: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    case = _load(config)
    parallel = case["parallel"]
    world_size = parallel["tp"] * parallel["cp"] * parallel["pp"] * parallel["dp"]
    command = [
        str(ROOT / "scripts/megatron_attention/run_case.sh"),
        str(config),
        str(world_size),
        str(output),
        adapter,
    ]
    started = time.perf_counter()
    returncode, timed_out = native_matrix._run_with_graceful_timeout(
        command,
        cwd=ROOT,
        timeout_seconds=timeout_seconds,
    )
    attempt = native_matrix._attempt_summary(
        output, returncode, time.perf_counter() - started
    )
    attempt.update(adapter=adapter, config=str(config), timed_out=timed_out)
    if timed_out:
        attempt.update(
            status="fail",
            error_type="TimeoutExpired",
            error="case exceeded timeout and was stopped with SIGINT/SIGTERM",
        )
    return attempt


def _compare_pair(
    native: dict[str, Any],
    magi: dict[str, Any],
    *,
    output: Path,
    output_atol: float,
    output_rtol: float,
    grad_atol: float,
    grad_rtol: float,
) -> dict[str, Any]:
    if native.get("status") != "pass" or magi.get("status") != "pass":
        return {"status": "not_run", "allclose": False, "reason": "backend run failed"}
    command = [
        str(ROOT / ".venv/bin/python"),
        str(ROOT / "scripts/megatron_attention/compare_golden.py"),
        native["golden_artifact"],
        magi["golden_artifact"],
        "--output",
        str(output),
        "--output-atol",
        str(output_atol),
        "--output-rtol",
        str(output_rtol),
        "--grad-atol",
        str(grad_atol),
        "--grad-rtol",
        str(grad_rtol),
    ]
    completed = subprocess.run(
        command, cwd=ROOT, check=False, capture_output=True, text=True
    )
    if not output.exists():
        return {
            "status": "fail",
            "allclose": False,
            "reason": "golden comparator produced no JSON",
            "returncode": completed.returncode,
            "stderr": completed.stderr,
        }
    result = _load(output)
    result["status"] = "pass" if completed.returncode == 0 and result["allclose"] else "fail"
    return result


def _summarize_pair(pair: dict[str, Any]) -> None:
    repeats = pair["repeats"]
    valid = [
        repeat
        for repeat in repeats
        if repeat["parity"].get("allclose")
        and repeat["native"].get("status") == "pass"
        and repeat["magi"].get("status") == "pass"
    ]
    if len(valid) != pair["repeat_count"]:
        pair["status"] = "fail"
        pair["performance_comparable"] = False
        return
    native_ms = [float(item["native"]["median_ms"]) for item in valid]
    magi_ms = [float(item["magi"]["median_ms"]) for item in valid]
    native_median = statistics.median(native_ms)
    magi_median = statistics.median(magi_ms)
    pair.update(
        status="pass",
        performance_comparable=True,
        native_median_ms=native_median,
        magi_median_ms=magi_median,
        speedup_native_over_magi=(native_median / magi_median),
    )


def main() -> int:
    args = _parser().parse_args()
    manifest = _load(args.manifest)
    selected = set(args.only)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "manifest": str(args.manifest),
        "purpose": manifest["purpose"],
        "timing_scope": "batch_materialize_through_backward_and_optimizer_zero",
        "started_unix_ns": time.time_ns(),
        "status": "running",
        "pairs": [],
    }
    repeat_count = int(manifest.get("repeats", 2))
    timeout_seconds = float(manifest.get("timeout_seconds", 300))
    for raw in manifest["pairs"]:
        pair_id = raw["pair_id"]
        if selected and pair_id not in selected:
            continue
        native_config = ROOT / raw["native_config"]
        magi_config = ROOT / raw["magi_config"]
        native_case = _load(native_config)
        magi_case = _load(magi_config)
        if native_case["harness_mode"] != magi_case["harness_mode"]:
            raise ValueError(f"{pair_id}: harness mismatch")
        if native_case["sequence"] != magi_case["sequence"]:
            raise ValueError(f"{pair_id}: sequence mismatch")
        pair = {
            "pair_id": pair_id,
            "harness_mode": native_case["harness_mode"],
            "attention_mode": native_case["attention_mode"],
            "native_config": str(native_config),
            "magi_config": str(magi_config),
            "repeat_count": repeat_count,
            "repeats": [],
        }
        summary["pairs"].append(pair)
        if args.dry_run:
            pair.update(status="dry_run", performance_comparable=False)
            continue
        correctness = native_case["correctness"]
        for repeat in range(repeat_count):
            order = ("native", "magi") if repeat % 2 == 0 else ("magi", "native")
            record: dict[str, Any] = {"repeat": repeat, "launch_order": list(order)}
            for adapter in order:
                config = native_config if adapter == "native" else magi_config
                output = args.output_dir / f"{pair_id}.repeat{repeat}.{adapter}.json"
                output.unlink(missing_ok=True)
                record[adapter] = _run_case(
                    config=config,
                    adapter=adapter,
                    output=output,
                    timeout_seconds=float(raw.get("timeout_seconds", timeout_seconds)),
                )
                native_matrix._atomic_json(summary, args.summary)
                if record[adapter]["status"] != "pass":
                    break
            record.setdefault("native", {"status": "not_run"})
            record.setdefault("magi", {"status": "not_run"})
            parity_output = args.output_dir / f"{pair_id}.repeat{repeat}.parity.json"
            record["parity"] = _compare_pair(
                record["native"],
                record["magi"],
                output=parity_output,
                output_atol=correctness["output_atol"],
                output_rtol=correctness["output_rtol"],
                grad_atol=correctness["grad_atol"],
                grad_rtol=correctness["grad_rtol"],
            )
            pair["repeats"].append(record)
            native_matrix._atomic_json(summary, args.summary)
        _summarize_pair(pair)
        native_matrix._atomic_json(summary, args.summary)
    summary["status"] = (
        "pass"
        if all(pair["status"] in {"pass", "dry_run"} for pair in summary["pairs"])
        else "fail"
    )
    summary["finished_unix_ns"] = time.time_ns()
    native_matrix._atomic_json(summary, args.summary)
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
