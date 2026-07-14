"""Sample aggregation that preserves rank-local evidence."""

from __future__ import annotations

import math
import statistics
from typing import Iterable, Mapping


def summarize_samples(values: Iterable[float]) -> dict[str, float | int | None]:
    samples = [float(value) for value in values]
    if not samples:
        return {"count": 0, "min_ms": None, "median_ms": None, "p90_ms": None, "max_ms": None, "mean_ms": None, "cv": None}
    ordered = sorted(samples)
    mean = statistics.fmean(ordered)
    std = statistics.pstdev(ordered) if len(ordered) > 1 else 0.0
    p90_index = min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)
    return {
        "count": len(ordered),
        "min_ms": ordered[0],
        "median_ms": statistics.median(ordered),
        "p90_ms": ordered[p90_index],
        "max_ms": ordered[-1],
        "mean_ms": mean,
        "cv": std / mean if mean else 0.0,
    }


def summarize_timeline(samples: Iterable[Mapping[str, float]]) -> dict[str, object]:
    materialized = list(samples)
    names = sorted({name for sample in materialized for name in sample})
    return {
        name: summarize_samples(sample[name] for sample in materialized if name in sample)
        for name in names
    }


def max_rank_timeline(local_spans_ms: Mapping[str, float]) -> dict[str, float]:
    """All-reduce each completed span with MAX, or return local values outside dist."""
    try:
        import torch
        import torch.distributed as dist
    except ImportError:
        return dict(local_spans_ms)
    if not dist.is_available() or not dist.is_initialized():
        return dict(local_spans_ms)
    names = sorted(local_spans_ms)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tensor = torch.tensor([local_spans_ms[name] for name in names], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return {name: float(value) for name, value in zip(names, tensor.cpu().tolist(), strict=True)}
