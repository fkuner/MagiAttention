"""Benchmark adapters sharing one runner contract."""

from .base import BaseBenchmarkAdapter

__all__ = ["BaseBenchmarkAdapter", "MegatronNativeAdapter"]


def __getattr__(name: str):
    if name == "MegatronNativeAdapter":
        from .native import MegatronNativeAdapter

        return MegatronNativeAdapter
    raise AttributeError(name)
