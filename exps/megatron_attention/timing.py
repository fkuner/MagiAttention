"""Rank-local benchmark timing with optional NVTX annotation."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class TimelineSample:
    spans_ms: dict[str, float] = field(default_factory=dict)

    def add(self, name: str, elapsed_ms: float) -> None:
        if name in self.spans_ms:
            raise RuntimeError(f"timeline span {name!r} was recorded more than once")
        self.spans_ms[name] = float(elapsed_ms)


class SpanRecorder:
    """CPU wall-clock spans covering host work and GPU launch/sync boundaries."""

    def __init__(self, *, enable_nvtx: bool = True) -> None:
        self.sample = TimelineSample()
        self.enable_nvtx = enable_nvtx

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        nvtx = _nvtx_if_available() if self.enable_nvtx else None
        if nvtx is not None:
            nvtx.range_push(name)
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000.0
            if nvtx is not None:
                nvtx.range_pop()
            self.sample.add(name, elapsed_ms)


def _nvtx_if_available():
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.nvtx
    except (ImportError, RuntimeError):
        pass
    return None
