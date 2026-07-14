#!/usr/bin/env python3
"""Merge resumable T07 summaries into one auditable Native matrix."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def merge_summaries(sources: list[tuple[Path, dict[str, Any]]]) -> dict[str, Any]:
    cases_by_id: dict[str, dict[str, Any]] = {}
    required_groups: list[str] = []
    for _, payload in sources:
        for group in payload.get("required_groups", []):
            if group not in required_groups:
                required_groups.append(group)
        for case in payload.get("cases", []):
            case_id = case["case_id"]
            if case_id in cases_by_id:
                raise ValueError(f"duplicate case_id across summaries: {case_id}")
            cases_by_id[case_id] = case

    cases = list(cases_by_id.values())
    selection: dict[str, dict[str, Any]] = {}
    for group in required_groups:
        candidates = [
            case
            for case in cases
            if case.get("selection_group") == group and case.get("status") == "pass"
        ]
        if not candidates:
            continue
        winner = min(candidates, key=lambda case: case["median_ms_across_repeats"])
        selection[group] = {
            "case_id": winner["case_id"],
            "cp_comm_type": winner["cp_comm_type"],
            "median_ms_across_repeats": winner["median_ms_across_repeats"],
            "candidate_count": len(candidates),
        }

    missing = [group for group in required_groups if group not in selection]
    return {
        "schema_version": 1,
        "status": "pass" if not missing else "fail",
        "claim_scope": "T07 Native capability matrix; timings are not headline results",
        "created_unix_ns": time.time_ns(),
        "sources": [str(path) for path, _ in sources],
        "required_groups": required_groups,
        "missing_required_groups": missing,
        "selection": selection,
        "cases": cases,
    }


def _render_report(matrix: dict[str, Any]) -> str:
    cases = matrix["cases"]
    counts: dict[str, int] = {}
    for case in cases:
        status = case.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1
    lines = [
        "# T07 Megatron Native Matrix",
        "",
        "> This is a correctness and capability matrix. Its 512-token timings are not headline performance results.",
        "",
        f"- Status: `{matrix['status']}`",
        f"- Cases: `{len(cases)}`",
        f"- Status counts: `{json.dumps(counts, sort_keys=True)}`",
        f"- Missing required groups: `{matrix['missing_required_groups']}`",
        "",
        "## Selected Native baselines",
        "",
        "| Group | Case | CP mode | Median E2E (ms) | Passing candidates |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for group, winner in matrix["selection"].items():
        lines.append(
            f"| `{group}` | `{winner['case_id']}` | `{winner['cp_comm_type']}` | "
            f"{winner['median_ms_across_repeats']:.6f} | {winner['candidate_count']} |"
        )
    lines.extend(
        [
            "",
            "## All cases",
            "",
            "| Case | Group | CP mode | Status | Repeat match | Median E2E (ms) | Max CV |",
            "| --- | --- | --- | --- | --- | ---: | ---: |",
        ]
    )
    for case in cases:
        median = case.get("median_ms_across_repeats")
        cv = case.get("max_intra_run_cv")
        lines.append(
            f"| `{case['case_id']}` | `{case.get('selection_group')}` | "
            f"`{case.get('cp_comm_type')}` | `{case.get('status')}` | "
            f"`{case.get('repeat_numerical_match')}` | "
            f"{median:.6f} | {cv:.6f} |"
            if median is not None and cv is not None
            else f"| `{case['case_id']}` | `{case.get('selection_group')}` | "
            f"`{case.get('cp_comm_type')}` | `{case.get('status')}` | n/a | n/a | n/a |"
        )
    unsupported = [case for case in cases if case.get("status") != "pass"]
    if unsupported:
        lines.extend(
            [
                "",
                "## Unsupported or failed candidates",
                "",
                "| Case | Error type | Reason |",
                "| --- | --- | --- |",
            ]
        )
        for case in unsupported:
            attempts = case.get("attempts", [])
            failed = next((attempt for attempt in attempts if attempt.get("status") != "pass"), {})
            reason = str(failed.get("error", "no structured error was recorded"))
            reason = " ".join(reason.split()).replace("|", "\\|")
            lines.append(
                f"| `{case['case_id']}` | `{failed.get('error_type', 'unknown')}` | {reason} |"
            )
    lines.extend(["", "## Sources", ""])
    lines.extend(f"- `{source}`" for source in matrix["sources"])
    lines.append("")
    return "\n".join(lines)


def _atomic_write(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def main() -> int:
    args = _parser().parse_args()
    sources = [(path, _load(path)) for path in args.source]
    matrix = merge_summaries(sources)
    _atomic_write(json.dumps(matrix, indent=2, sort_keys=True) + "\n", args.output)
    _atomic_write(_render_report(matrix), args.report)
    return 0 if matrix["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
