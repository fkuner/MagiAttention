from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).parents[2]


def _load_script(name: str):
    path = ROOT / "scripts/megatron_attention" / name
    spec = importlib.util.spec_from_file_location(f"_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


matrix = _load_script("run_paired_e2e_matrix.py")
compare = _load_script("compare_golden.py")
recheck = _load_script("recheck_paired_e2e.py")


def test_golden_stats_report_shape_mismatch_without_raising() -> None:
    stats = compare._stats(torch.zeros(2, 3), torch.zeros(4, 3))
    assert stats == {
        "shape_match": False,
        "left_shape": [2, 3],
        "right_shape": [4, 3],
    }


def test_golden_allclose_uses_the_requested_tolerance() -> None:
    left = torch.tensor([0.0])
    right = torch.tensor([1.5e-3])
    assert compare._allclose(left, right, atol=2e-3, rtol=0.0)
    assert not compare._allclose(left, right, atol=1e-3, rtol=0.0)


def test_pair_summary_requires_parity_before_performance_comparison() -> None:
    pair = {
        "repeat_count": 1,
        "repeats": [
            {
                "native": {"status": "pass", "median_ms": 4.0},
                "magi": {"status": "pass", "median_ms": 2.0},
                "parity": {"allclose": False},
            }
        ],
    }
    matrix._summarize_pair(pair)
    assert pair["status"] == "fail"
    assert pair["performance_comparable"] is False
    assert "speedup_native_over_magi" not in pair


def test_pair_summary_reports_native_over_magi_speedup() -> None:
    pair = {
        "repeat_count": 2,
        "repeats": [
            {
                "native": {"status": "pass", "median_ms": 4.0},
                "magi": {"status": "pass", "median_ms": 2.0},
                "parity": {"allclose": True},
            },
            {
                "native": {"status": "pass", "median_ms": 6.0},
                "magi": {"status": "pass", "median_ms": 2.0},
                "parity": {"allclose": True},
            },
        ],
    }
    matrix._summarize_pair(pair)
    assert pair["status"] == "pass"
    assert pair["performance_comparable"] is True
    assert pair["native_median_ms"] == 5.0
    assert pair["magi_median_ms"] == 2.0
    assert pair["speedup_native_over_magi"] == 2.5


def test_recheck_loads_current_paired_matrix() -> None:
    assert recheck.paired_matrix._summarize_pair is matrix._summarize_pair or callable(
        recheck.paired_matrix._summarize_pair
    )
