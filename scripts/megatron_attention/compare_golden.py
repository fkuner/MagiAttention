#!/usr/bin/env python3
"""Compare two benchmark golden artifacts tensor-by-tensor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _stats(left: torch.Tensor, right: torch.Tensor) -> dict[str, object]:
    if left.shape != right.shape:
        return {"shape_match": False, "left_shape": list(left.shape), "right_shape": list(right.shape)}
    lhs = left.float()
    rhs = right.float()
    absolute = (lhs - rhs).abs()
    denominator = torch.maximum(lhs.abs(), rhs.abs()).clamp_min(1e-8)
    return {
        "shape_match": True,
        "max_abs": float(absolute.max().item()) if absolute.numel() else 0.0,
        "mean_abs": float(absolute.mean().item()) if absolute.numel() else 0.0,
        "max_rel": float((absolute / denominator).max().item()) if absolute.numel() else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=2e-3)
    parser.add_argument("--rtol", type=float, default=2e-3)
    args = parser.parse_args()

    left = torch.load(args.left, map_location="cpu", weights_only=True)
    right = torch.load(args.right, map_location="cpu", weights_only=True)
    result: dict[str, object] = {
        "left": str(args.left),
        "right": str(args.right),
        "atol": args.atol,
        "rtol": args.rtol,
        "output": _stats(left["output"], right["output"]),
        "input_grad": _stats(left["input_grad"], right["input_grad"]),
    }
    left_grads = left["parameter_grads"]
    right_grads = right["parameter_grads"]
    result["parameter_keys_match"] = set(left_grads) == set(right_grads)
    result["parameter_grads"] = {
        name: _stats(left_grads[name], right_grads[name])
        for name in sorted(set(left_grads) & set(right_grads))
    }
    tensors = [
        (left["output"], right["output"]),
        (left["input_grad"], right["input_grad"]),
        *((left_grads[name], right_grads[name]) for name in sorted(set(left_grads) & set(right_grads))),
    ]
    result["allclose"] = bool(
        result["parameter_keys_match"]
        and all(torch.allclose(lhs.float(), rhs.float(), atol=args.atol, rtol=args.rtol) for lhs, rhs in tensors)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["allclose"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
