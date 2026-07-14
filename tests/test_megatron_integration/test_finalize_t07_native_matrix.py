from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_finalize_module():
    path = (
        Path(__file__).parents[2]
        / "scripts"
        / "megatron_attention"
        / "finalize_t07_native_matrix.py"
    )
    spec = importlib.util.spec_from_file_location("_finalize_t07_native_matrix", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


finalize = _load_finalize_module()


def _case(case_id: str, group: str, mode: str, latency: float, status: str = "pass"):
    return {
        "case_id": case_id,
        "selection_group": group,
        "cp_comm_type": mode,
        "status": status,
        "median_ms_across_repeats": latency,
    }


def test_merge_uses_fastest_case_across_resumed_summaries(tmp_path):
    sources = [
        (
            tmp_path / "first.json",
            {
                "required_groups": ["dense_cp2"],
                "cases": [
                    _case("p2p", "dense_cp2", "p2p", 3.0),
                    _case("a2a", "dense_cp2", "a2a", 5.0),
                ],
            },
        ),
        (
            tmp_path / "second.json",
            {
                "required_groups": ["dsa_cp2"],
                "cases": [_case("dsa", "dsa_cp2", "allgather", 7.0)],
            },
        ),
    ]

    matrix = finalize.merge_summaries(sources)

    assert matrix["status"] == "pass"
    assert matrix["selection"]["dense_cp2"]["case_id"] == "p2p"
    assert matrix["selection"]["dsa_cp2"]["case_id"] == "dsa"


def test_merge_fails_when_required_group_has_no_passing_case(tmp_path):
    matrix = finalize.merge_summaries(
        [
            (
                tmp_path / "summary.json",
                {
                    "required_groups": ["dense_cp4"],
                    "cases": [
                        _case(
                            "unsupported",
                            "dense_cp4",
                            "hierarchical",
                            0.0,
                            status="unsupported_or_failed",
                        )
                    ],
                },
            )
        ]
    )

    assert matrix["status"] == "fail"
    assert matrix["missing_required_groups"] == ["dense_cp4"]


def test_report_preserves_unsupported_reason():
    case = _case(
        "unsupported",
        "dense_cp4",
        "hierarchical",
        0.0,
        status="unsupported_or_failed",
    )
    case["attempts"] = [
        {
            "status": "fail",
            "error_type": "ValueError",
            "error": "hierarchical requires a valid 2D mesh",
        }
    ]
    report = finalize._render_report(
        {
            "status": "pass",
            "cases": [case],
            "selection": {},
            "missing_required_groups": [],
            "sources": [],
        }
    )

    assert "hierarchical requires a valid 2D mesh" in report
