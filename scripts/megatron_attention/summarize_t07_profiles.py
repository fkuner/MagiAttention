#!/usr/bin/env python3
"""Validate and summarize the three selected T07 Native nsys captures."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

EXPECTED_PREFIXES = (
    "dense_sbhd_cp4_all_gather",
    "dense_thd_cp4_p2p",
    "dsa_sbhd_cp4_allgather",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _range_instances(rows: list[dict[str, str]], name: str) -> int:
    for row in rows:
        if row.get("Range") == name:
            return int(row["Range Instances"])
    return 0


def summarize_capture(directory: Path, prefix: str) -> dict[str, Any]:
    result_path = directory / f"{prefix}.json"
    status_path = directory / f"{prefix}.status"
    report_path = directory / f"{prefix}.nsys-rep"
    kernel_path = directory / f"{prefix}.stats_cuda_gpu_kern_sum.csv"
    nvtx_path = directory / f"{prefix}.stats_nvtx_gpu_proj_sum.csv"
    required = (result_path, status_path, report_path, kernel_path, nvtx_path)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        return {"prefix": prefix, "status": "missing", "missing": missing}

    result = json.loads(result_path.read_text())
    exit_code = int(status_path.read_text().strip())
    nvtx_rows = _read_csv(nvtx_path)
    kernel_rows = _read_csv(kernel_path)
    expected_instances = int(result["case"]["iterations"]) * int(result["world_size"])
    forward_instances = _range_instances(nvtx_rows, ":forward")
    backward_instances = _range_instances(nvtx_rows, ":backward")
    capture_valid = (
        exit_code == 0
        and result.get("status") == "pass"
        and forward_instances == expected_instances
        and backward_instances == expected_instances
        and bool(kernel_rows)
    )
    top_kernels = []
    for row in kernel_rows[:5]:
        top_kernels.append(
            {
                "name": row["Name"],
                "time_percent": float(row["Time (%)"]),
                "total_time_ns": int(row["Total Time (ns)"]),
                "instances": int(row["Instances"]),
                "median_ns": float(row["Med (ns)"]),
                "max_ns": int(row["Max (ns)"]),
            }
        )
    return {
        "prefix": prefix,
        "status": "pass" if capture_valid else "fail",
        "case_name": result["case"]["case_name"],
        "exit_code": exit_code,
        "expected_measured_range_instances": expected_instances,
        "forward_range_instances": forward_instances,
        "backward_range_instances": backward_instances,
        "profiled_headline_median_ms": result["summary"]["headline_e2e"]["median_ms"],
        "profiled_headline_cv": result["summary"]["headline_e2e"]["cv"],
        "top_kernels": top_kernels,
        "artifacts": {path.name: path.stat().st_size for path in required},
    }


def summarize_directory(directory: Path) -> dict[str, Any]:
    captures = [summarize_capture(directory, prefix) for prefix in EXPECTED_PREFIXES]
    return {
        "schema_version": 1,
        "status": "pass" if all(item["status"] == "pass" for item in captures) else "fail",
        "claim_scope": (
            "T07 diagnostic nsys captures; profiler-perturbed timings are not headline results"
        ),
        "captures": captures,
    }


def _render(payload: dict[str, Any]) -> str:
    lines = [
        "# T07 Native Nsight Systems Profiles",
        "",
        "> Diagnostic captures only. Profiled timings are perturbed and are not headline results.",
        "",
        f"- Status: `{payload['status']}`",
        "",
        "| Case | Status | Expected ranges | Forward | Backward | Profiled median (ms) | CV |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for capture in payload["captures"]:
        if capture["status"] == "missing":
            lines.append(f"| `{capture['prefix']}` | `missing` | n/a | n/a | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| `{capture['case_name']}` | `{capture['status']}` | "
            f"{capture['expected_measured_range_instances']} | "
            f"{capture['forward_range_instances']} | {capture['backward_range_instances']} | "
            f"{capture['profiled_headline_median_ms']:.6f} | "
            f"{capture['profiled_headline_cv']:.6f} |"
        )
    for capture in payload["captures"]:
        if not capture.get("top_kernels"):
            continue
        lines.extend(["", f"## {capture['case_name']} top kernels", "", "| Time | Instances | Median ns | Max ns | Kernel |", "| ---: | ---: | ---: | ---: | --- |"])
        for kernel in capture["top_kernels"]:
            name = kernel["name"].replace("|", "\\|")
            lines.append(
                f"| {kernel['time_percent']:.1f}% | {kernel['instances']} | "
                f"{kernel['median_ns']:.1f} | {kernel['max_ns']} | `{name}` |"
            )
    lines.append("")
    return "\n".join(lines)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    payload = summarize_directory(args.directory)
    _atomic_write(args.output, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _atomic_write(args.report, _render(payload))
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
