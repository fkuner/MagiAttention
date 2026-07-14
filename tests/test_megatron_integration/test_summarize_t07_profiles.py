from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "megatron_attention" / "summarize_t07_profiles.py"
SPEC = importlib.util.spec_from_file_location("summarize_t07_profiles", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_capture_requires_exact_measured_range_count(tmp_path: Path) -> None:
    prefix = "dense_sbhd_cp4_all_gather"
    (tmp_path / f"{prefix}.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "world_size": 4,
                "case": {"case_name": prefix, "iterations": 3},
                "summary": {"headline_e2e": {"median_ms": 7.0, "cv": 0.1}},
            }
        )
    )
    (tmp_path / f"{prefix}.status").write_text("0\n")
    (tmp_path / f"{prefix}.nsys-rep").write_bytes(b"report")
    (tmp_path / f"{prefix}.stats_cuda_gpu_kern_sum.csv").write_text(
        "Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name\n"
        "100.0,1200,12,100,100,90,110,1,kernel\n"
    )
    nvtx = tmp_path / f"{prefix}.stats_nvtx_gpu_proj_sum.csv"
    header = (
        "Range,Style,Total Proj Time (ns),Total Range Time (ns),Range Instances,"
        "Proj Avg (ns),Proj Med (ns),Proj Min (ns),Proj Max (ns),Proj StdDev (ns),"
        "Total GPU Ops,Avg GPU Ops,Avg Range Lvl,Avg Num Child\n"
    )
    nvtx.write_text(
        header
        + ":forward,PushPop,1,1,12,1,1,1,1,0,12,1,0,0\n"
        + ":backward,PushPop,1,1,12,1,1,1,1,0,12,1,0,0\n"
    )

    capture = MODULE.summarize_capture(tmp_path, prefix)
    assert capture["status"] == "pass"
    assert capture["expected_measured_range_instances"] == 12
    assert capture["top_kernels"][0]["name"] == "kernel"

    nvtx.write_text(
        header
        + ":forward,PushPop,1,1,11,1,1,1,1,0,11,1,0,0\n"
        + ":backward,PushPop,1,1,12,1,1,1,1,0,12,1,0,0\n"
    )
    assert MODULE.summarize_capture(tmp_path, prefix)["status"] == "fail"
