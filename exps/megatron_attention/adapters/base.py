"""Common adapter lifecycle; backend-specific logic only implements hooks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

from ..batch import GlobalBatch
from ..timing import SpanRecorder


class BaseBenchmarkAdapter(ABC):
    """One orchestration shape for Native and Magi implementations."""

    def __init__(self) -> None:
        self._recorder: SpanRecorder | None = None

    @property
    @abstractmethod
    def name(self) -> str: ...

    def begin_step(self, recorder: SpanRecorder) -> None:
        self._recorder = recorder

    def prepare_batch(self, global_batch: GlobalBatch, case: object) -> object:
        if self._recorder is None:
            raise RuntimeError("begin_step must be called before prepare_batch")
        with self._recorder.span("thd_pack"):
            packed = self.pack_batch(global_batch, case)
        with self._recorder.span("layout_plan"):
            plan = self.plan_layout(packed, case)
        with self._recorder.span("dispatch"):
            return self.dispatch_batch(packed, plan, case)

    def pack_batch(self, global_batch: GlobalBatch, case: object) -> object:
        return global_batch

    def plan_layout(self, packed_batch: object, case: object) -> object | None:
        return None

    @abstractmethod
    def dispatch_batch(self, packed_batch: object, plan: object | None, case: object) -> object: ...

    @abstractmethod
    def initialize_process_groups(self, case: object) -> object: ...

    @abstractmethod
    def build(self, case: object, process_groups: object, shared_state: object) -> object: ...

    @abstractmethod
    def forward(self, modules: object, prepared_batch: object, case: object) -> object: ...

    @abstractmethod
    def compute_loss(self, output: object, prepared_batch: object, case: object) -> object: ...

    @abstractmethod
    def backward(self, loss: object, modules: object, case: object) -> None: ...

    @abstractmethod
    def zero_grad(self, modules: object, case: object) -> None: ...

    @abstractmethod
    def canonicalize_for_check(self, output: object, prepared_batch: object) -> object: ...

    @abstractmethod
    def communication_counters(self) -> Mapping[str, object]: ...

    def synchronize(self) -> None:
        """Synchronize backend work before closing an end-to-end sample."""

    def close(self) -> None:
        """Release resources; implementations must not destroy shared user state."""
