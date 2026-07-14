#!/usr/bin/env python3
"""Generate T07 one-layer-training cases from proven Native probe winners.

The capability probe enumerates every advertised Native CP communication mode.
This manifest deliberately carries only the fastest correct mode in each group
into the more expensive local-LM-loss harness.
"""

from __future__ import annotations

import copy
import json

from generate_t07_probe_cases import ROOT, _case

CASE_DIR = ROOT / "exps/megatron_attention/cases/t07_training"
MANIFEST = ROOT / "exps/megatron_attention/t07_native_training_manifest.json"


# Frozen from the two-repeat B200 capability probe.  These timings are only
# used to select a correct Native mode; they are not headline performance data.
WINNERS = (
    (1, "none", "sbhd", False),
    (2, "p2p", "sbhd", False),
    (4, "all_gather", "sbhd", False),
    (1, "none", "thd", False),
    (2, "p2p", "thd", False),
    (4, "p2p", "thd", False),
    (1, "none", "sbhd", True),
    (2, "allgather", "sbhd", True),
    (4, "allgather", "sbhd", True),
)


def _training_case(*, cp: int, mode: str, layout: str, dsa: bool) -> dict:
    case = copy.deepcopy(_case(cp=cp, mode=mode, layout=layout, dsa=dsa))
    kind = "dsa" if dsa else "dense"
    case["case_name"] = f"t07_training_{kind}_{layout}_cp{cp}_{mode}"
    case["harness_mode"] = "attention_to_loss"
    case["loss_policy"] = "local_lm_loss"
    capabilities = case["backend"]["capabilities"]
    capabilities[capabilities.index("fixed_gradient")] = "local_lm_loss"
    case["claim_scope"] = (
        "T07 one-layer-training capability probe with local LM loss; "
        "timing is not a headline result"
    )
    return case


def main() -> None:
    specs = []
    required_groups = []
    for cp, mode, layout, dsa in WINNERS:
        case = _training_case(cp=cp, mode=mode, layout=layout, dsa=dsa)
        path = CASE_DIR / f"{case['case_name']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(case, indent=2) + "\n")
        kind = "dsa" if dsa else "dense"
        group = f"training_{kind}_{layout}_cp{cp}"
        required_groups.append(group)
        specs.append(
            {
                "case_id": case["case_name"],
                "config": str(path.relative_to(ROOT)),
                "required": True,
                "selection_group": group,
            }
        )

    MANIFEST.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "repeats": 2,
                "timeout_seconds": 240,
                "purpose": (
                    "T07 one-layer-training Native capability probe using "
                    "the fastest correct modes selected by the attention-stack probe"
                ),
                "required_groups": required_groups,
                "cases": specs,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
