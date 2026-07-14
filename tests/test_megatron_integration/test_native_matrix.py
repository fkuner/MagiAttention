from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_matrix_module():
    path = (
        Path(__file__).parents[2]
        / "scripts"
        / "megatron_attention"
        / "run_native_matrix.py"
    )
    spec = importlib.util.spec_from_file_location("_native_matrix", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


matrix = _load_matrix_module()


def test_graceful_timeout_uses_interrupt_without_hanging(tmp_path):
    returncode, timed_out = matrix._run_with_graceful_timeout(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        cwd=tmp_path,
        timeout_seconds=0.05,
        grace_seconds=1.0,
    )

    assert timed_out is True
    assert returncode != 0


def test_finalize_entry_requires_repeatable_golden_values():
    entry = {
        "repeats": 2,
        "correctness": {
            "output_atol": 0.001,
            "output_rtol": 0.001,
            "grad_atol": 0.002,
            "grad_rtol": 0.002,
        },
        "attempts": [
            {
                "status": "pass",
                "median_ms": 2.0,
                "cv": 0.02,
                "output_sha256": "o",
                "input_grad_sha256": "i",
                "state_sha256": {"p": "s"},
                "parameter_grad_sha256": {"p": "g"},
                "loss": 1.0,
            },
            {
                "status": "pass",
                "median_ms": 3.0,
                "cv": 0.03,
                "output_sha256": "o",
                "input_grad_sha256": "i",
                "state_sha256": {"p": "s"},
                "parameter_grad_sha256": {"p": "g"},
                "loss": 1.0,
            },
        ],
    }

    matrix._finalize_entry(entry)

    assert entry["status"] == "pass"
    assert entry["repeat_golden_match"] is True
    assert entry["repeat_numerical_match"] is True
    assert entry["median_ms_across_repeats"] == 2.5


def test_selection_uses_fastest_passing_candidate():
    entries = [
        {
            "case_id": "slow",
            "selection_group": "g",
            "cp_comm_type": "all_gather",
            "status": "pass",
            "median_ms_across_repeats": 4.0,
        },
        {
            "case_id": "fast",
            "selection_group": "g",
            "cp_comm_type": "p2p",
            "status": "pass",
            "median_ms_across_repeats": 2.0,
        },
        {
            "case_id": "failed",
            "selection_group": "g",
            "cp_comm_type": "a2a",
            "status": "unsupported_or_failed",
        },
    ]

    selected = matrix._selection_table(entries)

    assert selected["g"]["case_id"] == "fast"
    assert selected["g"]["candidate_count"] == 2
