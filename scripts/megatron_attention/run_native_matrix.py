#!/usr/bin/env python3
"""Run a repeatable Megatron-native capability/golden matrix.

The script deliberately launches each case in a fresh torchrun process so
process groups, kernel selection, and allocator state cannot leak across cases.
It writes the summary after every attempt, making unsupported combinations and
partial progress auditable.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _run_with_graceful_timeout(
    command: list[str], *, cwd: Path, timeout_seconds: float, grace_seconds: float = 30.0
) -> tuple[int, bool]:
    """Run one isolated torchrun and stop a hang without ever using SIGKILL."""

    process = subprocess.Popen(command, cwd=cwd, start_new_session=True)
    try:
        return process.wait(timeout=timeout_seconds), False
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGINT)
        try:
            return process.wait(timeout=grace_seconds), True
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                return process.wait(timeout=grace_seconds), True
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"process group {process.pid} did not exit after SIGINT/SIGTERM"
                ) from exc


def _attempt_summary(output: Path, returncode: int, elapsed: float) -> dict[str, Any]:
    attempt: dict[str, Any] = {
        "output": str(output),
        "returncode": returncode,
        "launcher_wall_seconds": elapsed,
    }
    if not output.exists():
        attempt.update(status="fail", error="runner did not produce rank-0 JSON")
        return attempt
    result = _load(output)
    attempt["status"] = result.get("status", "fail")
    if result.get("status") == "pass":
        headline = result["summary"]["headline_e2e"]
        golden = result.get("correctness", {}).get("golden", {})
        attempt.update(
            median_ms=headline["median_ms"],
            cv=headline["cv"],
            output_sha256=golden.get("output_sha256"),
            input_grad_sha256=golden.get("input_grad_sha256"),
            state_sha256=golden.get("state_sha256"),
            parameter_grad_sha256=golden.get("parameter_grad_sha256"),
            loss=golden.get("loss"),
            golden_artifact=golden.get("artifact"),
        )
    else:
        attempt.update(
            error_type=result.get("error_type"),
            error=result.get("error"),
        )
    return attempt


def _finalize_entry(entry: dict[str, Any]) -> None:
    passed = [attempt for attempt in entry["attempts"] if attempt["status"] == "pass"]
    if len(passed) != entry["repeats"]:
        entry["status"] = "unsupported_or_failed"
        return
    golden_fields = (
        "output_sha256",
        "input_grad_sha256",
        "state_sha256",
        "parameter_grad_sha256",
        "loss",
    )
    entry["repeat_golden_match"] = all(
        all(attempt.get(field) == passed[0].get(field) for attempt in passed[1:])
        for field in golden_fields
    )
    entry["repeat_numerical_match"] = entry["repeat_golden_match"]
    if not entry["repeat_golden_match"]:
        entry["repeat_numerical_match"], entry["repeat_numerical_diffs"] = (
            _compare_repeat_artifacts(passed, entry["correctness"])
        )
    medians = [float(attempt["median_ms"]) for attempt in passed]
    entry["median_ms_across_repeats"] = statistics.median(medians)
    entry["max_intra_run_cv"] = max(float(attempt["cv"]) for attempt in passed)
    entry["status"] = "pass" if entry["repeat_numerical_match"] else "nondeterministic"


def _compare_repeat_artifacts(
    attempts: list[dict[str, Any]], correctness: dict[str, float]
) -> tuple[bool, list[dict[str, Any]]]:
    import torch

    if any(attempt.get("state_sha256") != attempts[0].get("state_sha256") for attempt in attempts):
        return False, [{"field": "state", "reason": "state hash differs"}]
    try:
        payloads = [torch.load(attempt["golden_artifact"], map_location="cpu") for attempt in attempts]
    except (OSError, KeyError) as exc:
        return False, [{"field": "artifact", "reason": str(exc)}]
    reference = payloads[0]
    comparisons = (
        ("output", correctness["output_atol"], correctness["output_rtol"]),
        ("input_grad", correctness["grad_atol"], correctness["grad_rtol"]),
    )
    diffs: list[dict[str, Any]] = []

    def compare(name: str, left, right, atol: float, rtol: float) -> bool:
        delta = (left.float() - right.float()).abs()
        max_abs = float(delta.max().item()) if delta.numel() else 0.0
        denominator = right.float().abs().clamp_min(1e-12)
        max_rel = float((delta / denominator).max().item()) if delta.numel() else 0.0
        match = bool(torch.allclose(left.float(), right.float(), atol=atol, rtol=rtol))
        diffs.append(
            {"field": name, "allclose": match, "max_abs": max_abs, "max_rel": max_rel}
        )
        return match

    matches = []
    for attempt, payload in zip(attempts[1:], payloads[1:], strict=True):
        for name, atol, rtol in comparisons:
            matches.append(compare(name, payload[name], reference[name], atol, rtol))
        reference_grads = reference.get("parameter_grads", {})
        candidate_grads = payload.get("parameter_grads", {})
        if reference_grads.keys() != candidate_grads.keys() or not reference_grads:
            matches.append(False)
            diffs.append({"field": "parameter_grads", "reason": "missing or mismatched keys"})
        else:
            for name in reference_grads:
                matches.append(
                    compare(
                        f"parameter_grads.{name}",
                        candidate_grads[name],
                        reference_grads[name],
                        correctness["grad_atol"],
                        correctness["grad_rtol"],
                    )
                )
        loss_delta = abs(float(attempt["loss"]) - float(attempts[0]["loss"]))
        loss_match = loss_delta <= correctness["output_atol"]
        matches.append(loss_match)
        diffs.append({"field": "loss", "allclose": loss_match, "max_abs": loss_delta})
    return all(matches), diffs


def _selection_table(entries: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        group = entry.get("selection_group")
        if group and entry.get("status") == "pass":
            groups.setdefault(group, []).append(entry)
    selection: dict[str, Any] = {}
    for group, candidates in groups.items():
        winner = min(candidates, key=lambda item: item["median_ms_across_repeats"])
        selection[group] = {
            "case_id": winner["case_id"],
            "cp_comm_type": winner["cp_comm_type"],
            "median_ms_across_repeats": winner["median_ms_across_repeats"],
            "candidate_count": len(candidates),
        }
    return selection


def main() -> int:
    args = _parser().parse_args()
    manifest = _load(args.manifest)
    project_root = Path(__file__).resolve().parents[2]
    selected = set(args.only)
    raw_cases = manifest.get("cases", [])
    if not isinstance(raw_cases, list):
        raise ValueError("manifest.cases must be a list")
    required_groups = list(manifest.get("required_groups", []))
    if selected:
        selected_groups = {
            raw.get("selection_group")
            for raw in raw_cases
            if raw.get("case_id") in selected and raw.get("selection_group") is not None
        }
        required_groups = [group for group in required_groups if group in selected_groups]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "manifest": str(args.manifest),
        "started_unix_ns": time.time_ns(),
        "status": "running",
        "cases": [],
        "selection": {},
        "required_groups": required_groups,
    }
    for raw in raw_cases:
        case_id = raw["case_id"]
        if selected and case_id not in selected:
            continue
        config_path = project_root / raw["config"]
        config = _load(config_path)
        repeats = int(raw.get("repeats", manifest.get("repeats", 2)))
        entry = {
            "case_id": case_id,
            "config": str(config_path),
            "selection_group": raw.get("selection_group"),
            "required": bool(raw.get("required", False)),
            "correctness": config["correctness"],
            "cp_comm_type": config["backend"]["cp_comm_type"],
            "world_size": config["parallel"]["tp"]
            * config["parallel"]["cp"]
            * config["parallel"]["pp"]
            * config["parallel"]["dp"],
            "repeats": repeats,
            "attempts": [],
        }
        summary["cases"].append(entry)
        if args.dry_run:
            entry["status"] = "dry_run"
            continue
        for repeat in range(repeats):
            output = args.output_dir / f"{case_id}.repeat{repeat}.json"
            output.unlink(missing_ok=True)
            command = [
                str(project_root / "scripts/megatron_attention/run_case.sh"),
                str(config_path),
                str(entry["world_size"]),
                str(output),
            ]
            started = time.perf_counter()
            returncode, timed_out = _run_with_graceful_timeout(
                command,
                cwd=project_root,
                timeout_seconds=float(raw.get("timeout_seconds", manifest.get("timeout_seconds", 300))),
            )
            attempt = _attempt_summary(output, returncode, time.perf_counter() - started)
            if timed_out:
                attempt.update(
                    status="fail",
                    error_type="TimeoutExpired",
                    error="case exceeded its timeout and was stopped with SIGINT/SIGTERM",
                )
            entry["attempts"].append(attempt)
            _atomic_json(summary, args.summary)
            if returncode and raw.get("stop_repeats_after_failure", True):
                break
        _finalize_entry(entry)
        summary["selection"] = _selection_table(summary["cases"])
        _atomic_json(summary, args.summary)
    missing_required_groups = (
        []
        if args.dry_run
        else sorted(set(summary["required_groups"]) - set(_selection_table(summary["cases"])))
    )
    summary["missing_required_groups"] = missing_required_groups
    summary["status"] = (
        "pass"
        if all(
            entry["status"] in {"pass", "dry_run", "unsupported_or_failed"}
            and (not entry["required"] or entry["status"] in {"pass", "dry_run"})
            for entry in summary["cases"]
        )
        and not missing_required_groups
        else "fail"
    )
    summary["finished_unix_ns"] = time.time_ns()
    summary["selection"] = _selection_table(summary["cases"])
    _atomic_json(summary, args.summary)
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
