"""Single entry point for Megatron attention benchmark cases."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Sequence

from .batch import BatchFactory
from .config import AttentionBenchmarkCase, ConfigError, load_case
from .metrics import max_rank_timeline, summarize_timeline
from .provenance import collect_provenance
from .timing import SpanRecorder


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-effective-config", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--adapter",
        choices=("native", "magi"),
        help="Concrete execution adapter. Required unless --dry-run is used.",
    )
    return parser


class BenchmarkRunner:
    PHASES = (
        "validate",
        "init_groups",
        "materialize",
        "build",
        "correctness",
        "warmup",
        "measure",
        "report",
    )

    def __init__(
        self,
        case: AttentionBenchmarkCase,
        backend,
        *,
        output_path: Path,
        repo_root: Path,
        batch_factory: BatchFactory | None = None,
    ) -> None:
        self.case = case
        self.backend = backend
        self.output_path = output_path
        self.repo_root = repo_root
        self.batch_factory = batch_factory or BatchFactory()
        self.completed_phases: list[str] = []

    def run(self) -> dict[str, object]:
        started_ns = time.time_ns()
        result: dict[str, object] = {
            "schema_version": 1,
            "status": "running",
            "case": self.case.to_dict(),
            "backend": self.backend.name,
            "completed_phases": self.completed_phases,
            "provenance": collect_provenance(self.repo_root),
            "started_unix_ns": started_ns,
            "rank": _distributed_rank(),
            "world_size": _distributed_world_size(),
        }
        try:
            self.case.validate()
            self._complete("validate")
            process_groups = self.backend.initialize_process_groups(self.case)
            self._complete("init_groups")
            global_batch = self.batch_factory.build(self.case)
            self._complete("materialize")
            modules = self.backend.build(self.case, process_groups, shared_state=None)
            self._complete("build")
            correctness = self._correctness_step(global_batch, modules)
            self._complete("correctness")
            for _ in range(self.case.warmup):
                self._iteration(modules, measure=False)
            self._complete("warmup")
            profiler_started = False
            if _cuda_profiler_range_enabled():
                _cuda_profiler_start()
                profiler_started = True
            try:
                rank_local_samples = [
                    self._iteration(modules, measure=True) for _ in range(self.case.iterations)
                ]
            finally:
                if profiler_started:
                    _cuda_profiler_stop()
            max_rank_samples = [max_rank_timeline(sample) for sample in rank_local_samples]
            self._complete("measure")
            result.update(
                {
                    "status": "pass",
                    "correctness": correctness,
                    "rank_local_samples_ms": rank_local_samples,
                    "max_rank_samples_ms": max_rank_samples,
                    "summary": summarize_timeline(max_rank_samples),
                    "communication": dict(self.backend.communication_counters()),
                    "finished_unix_ns": time.time_ns(),
                }
            )
            self._complete("report")
            self._write(result)
            return result
        except Exception as exc:
            result.update(
                {
                    "status": "fail",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "finished_unix_ns": time.time_ns(),
                }
            )
            self._write(result)
            raise
        finally:
            self.backend.close()

    def _correctness_step(self, global_batch, modules) -> dict[str, object]:
        recorder = SpanRecorder(enable_nvtx=False)
        self.backend.begin_step(recorder)
        prepared = self.backend.prepare_batch(global_batch, self.case)
        output = self.backend.forward(modules, prepared, self.case)
        canonical = self.backend.canonicalize_for_check(output, prepared)
        self.backend.zero_grad(modules, self.case)
        metadata = {
            "status": "structural_pass",
            "canonical_type": type(canonical).__name__,
        }
        golden_hook = getattr(self.backend, "capture_golden", None)
        if golden_hook is not None:
            golden = dict(golden_hook(modules, prepared, self.case))
            tensors = golden.pop("_tensors", None)
            if tensors is not None and _distributed_rank() == 0:
                artifact = self.output_path.with_name(
                    f"{self.output_path.stem}.golden.pt"
                )
                _atomic_torch_save(tensors, artifact)
                golden["artifact"] = str(artifact)
            metadata["golden"] = golden
        regression_hook = getattr(self.backend, "run_correctness_regressions", None)
        if regression_hook is not None:
            metadata.update(regression_hook(modules, prepared, self.case))
        hook = getattr(self.backend, "correctness_metadata", None)
        if hook is not None:
            metadata.update(hook(canonical, modules, prepared, self.case))
        return metadata

    def _iteration(self, modules, *, measure: bool) -> dict[str, float]:
        recorder = SpanRecorder(enable_nvtx=measure)
        self.backend.begin_step(recorder)
        headline_start = time.perf_counter_ns()
        with recorder.span("batch_materialize"):
            global_batch = self.batch_factory.build(self.case)
        prepared = self.backend.prepare_batch(global_batch, self.case)
        with recorder.span("forward"):
            output = self.backend.forward(modules, prepared, self.case)
        with recorder.span("local_loss"):
            loss = self.backend.compute_loss(output, prepared, self.case)
        with recorder.span("backward"):
            self.backend.backward(loss, modules, self.case)
        with recorder.span("optimizer_zero"):
            self.backend.zero_grad(modules, self.case)
        self.backend.synchronize()
        recorder.sample.add("headline_e2e", (time.perf_counter_ns() - headline_start) / 1_000_000.0)
        return recorder.sample.spans_ms

    def _complete(self, phase: str) -> None:
        expected = self.PHASES[len(self.completed_phases)]
        if phase != expected:
            raise RuntimeError(f"phase order violation: got {phase}, expected {expected}")
        self.completed_phases.append(phase)

    def _write(self, result: dict[str, object]) -> None:
        output_path = self.output_path
        rank = _distributed_rank()
        if rank:
            output_path = output_path.with_name(
                f"{output_path.stem}.rank{rank}{output_path.suffix}"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, output_path)


def _distributed_rank() -> int:
    try:
        import torch.distributed as dist
    except ImportError:
        return int(os.environ.get("RANK", "0"))
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", "0"))


def _distributed_world_size() -> int:
    try:
        import torch.distributed as dist
    except ImportError:
        return int(os.environ.get("WORLD_SIZE", "1"))
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", "1"))


def _atomic_torch_save(payload: object, output_path: Path) -> None:
    import torch

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output_path)


def _cuda_profiler_range_enabled() -> bool:
    return os.environ.get("MAGI_BENCH_CUDA_PROFILER_RANGE", "0") == "1"


def _cuda_profiler_start() -> None:
    import torch

    torch.cuda.synchronize()
    _distributed_barrier()
    torch.cuda.cudart().cudaProfilerStart()


def _cuda_profiler_stop() -> None:
    import torch

    torch.cuda.synchronize()
    _distributed_barrier()
    torch.cuda.cudart().cudaProfilerStop()


def _distributed_barrier() -> None:
    import torch
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        kwargs = {}
        if dist.get_backend() == "nccl":
            kwargs["device_ids"] = [torch.cuda.current_device()]
        dist.barrier(**kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        case = load_case(args.config)
    except ConfigError as exc:
        raise SystemExit(f"configuration error: {exc}") from exc
    if args.print_effective_config:
        print(json.dumps(case.to_dict(), indent=2, sort_keys=True))
    if args.dry_run:
        return 0
    if args.adapter is None:
        raise SystemExit("--adapter is required for execution")
    if args.output is None:
        raise SystemExit("--output is required for execution")
    from .adapters.magi import MagiExpandedMLAAdapter
    from .adapters.native import MegatronNativeAdapter

    adapter = MegatronNativeAdapter() if args.adapter == "native" else MagiExpandedMLAAdapter()

    runner = BenchmarkRunner(
        case,
        adapter,
        output_path=args.output,
        repo_root=Path(__file__).resolve().parents[2],
    )
    runner.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
