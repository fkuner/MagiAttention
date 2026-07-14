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


def _allclose(left: torch.Tensor, right: torch.Tensor, *, atol: float, rtol: float) -> bool:
    return bool(
        left.shape == right.shape
        and torch.allclose(left.float(), right.float(), atol=atol, rtol=rtol)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=2e-3)
    parser.add_argument("--rtol", type=float, default=2e-3)
    parser.add_argument("--output-atol", type=float)
    parser.add_argument("--output-rtol", type=float)
    parser.add_argument("--grad-atol", type=float)
    parser.add_argument("--grad-rtol", type=float)
    args = parser.parse_args()

    output_atol = args.atol if args.output_atol is None else args.output_atol
    output_rtol = args.rtol if args.output_rtol is None else args.output_rtol
    grad_atol = args.atol if args.grad_atol is None else args.grad_atol
    grad_rtol = args.rtol if args.grad_rtol is None else args.grad_rtol

    left = torch.load(args.left, map_location="cpu", weights_only=True)
    right = torch.load(args.right, map_location="cpu", weights_only=True)
    result: dict[str, object] = {
        "left": str(args.left),
        "right": str(args.right),
        "tolerances": {
            "output_atol": output_atol,
            "output_rtol": output_rtol,
            "grad_atol": grad_atol,
            "grad_rtol": grad_rtol,
        },
        "output": _stats(left["output"], right["output"]),
        "input_grad": _stats(left["input_grad"], right["input_grad"]),
    }
    optional_tensors = ("indexer_loss", "topk_global_ids", "topk_scores")
    result["optional_keys_match"] = all((name in left) == (name in right) for name in optional_tensors)
    for name in optional_tensors:
        if name in left and name in right:
            result[name] = _stats(left[name], right[name])
    left_grads = left["parameter_grads"]
    right_grads = right["parameter_grads"]
    result["parameter_keys_match"] = set(left_grads) == set(right_grads)
    result["parameter_grads"] = {
        name: _stats(left_grads[name], right_grads[name])
        for name in sorted(set(left_grads) & set(right_grads))
    }
    output_tensors = [
        (left["output"], right["output"]),
        *((left[name], right[name]) for name in optional_tensors if name in left and name in right),
    ]
    gradient_tensors = [
        (left["input_grad"], right["input_grad"]),
        *((left_grads[name], right_grads[name]) for name in sorted(set(left_grads) & set(right_grads))),
    ]
    result["output_allclose"] = all(
        _allclose(lhs, rhs, atol=output_atol, rtol=output_rtol)
        for lhs, rhs in output_tensors
    )
    result["grad_allclose"] = all(
        _allclose(lhs, rhs, atol=grad_atol, rtol=grad_rtol)
        for lhs, rhs in gradient_tensors
    )
    result["allclose"] = bool(
        result["parameter_keys_match"]
        and result["optional_keys_match"]
        and result["output_allclose"]
        and result["grad_allclose"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["allclose"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
