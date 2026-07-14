"""Benchmark adapters sharing one runner contract."""

from .base import BaseBenchmarkAdapter

__all__ = ["BaseBenchmarkAdapter", "MagiExpandedMLAAdapter", "MegatronNativeAdapter"]


def __getattr__(name: str):
    if name == "MegatronNativeAdapter":
        from .native import MegatronNativeAdapter

        return MegatronNativeAdapter
    if name == "MagiExpandedMLAAdapter":
        from .magi import MagiExpandedMLAAdapter

        return MagiExpandedMLAAdapter
    raise AttributeError(name)
