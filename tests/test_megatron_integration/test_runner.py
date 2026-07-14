from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import exps.megatron_attention.runner as runner_module
from exps.megatron_attention.adapters.base import BaseBenchmarkAdapter
from exps.megatron_attention.batch import GlobalBatch
from exps.megatron_attention.config import load_case
from exps.megatron_attention.runner import BenchmarkRunner

ROOT = Path(__file__).parents[2]
CASES = ROOT / "exps" / "megatron_attention" / "cases"


class MockAdapter(BaseBenchmarkAdapter):
    name = "mock"

    def __init__(self, *, fail_forward: bool = False) -> None:
        super().__init__()
        self.fail_forward = fail_forward
        self.closed = False

    def initialize_process_groups(self, case):
        return {"world_size": 1}

    def build(self, case, process_groups, shared_state):
        return torch.nn.Parameter(torch.tensor(2.0))

    def pack_batch(self, global_batch: GlobalBatch, case):
        return global_batch.clone()

    def plan_layout(self, packed_batch, case):
        return torch.arange(packed_batch.physical_num_tokens)

    def dispatch_batch(self, packed_batch, plan, case):
        return {"batch": packed_batch, "physical_ids": plan}

    def forward(self, modules, prepared_batch, case):
        if self.fail_forward:
            raise RuntimeError("intentional mock failure")
        tokens = prepared_batch["batch"].tokens.to(dtype=torch.float32)
        return tokens * modules

    def compute_loss(self, output, prepared_batch, case):
        mask = prepared_batch["batch"].loss_mask
        return (output * mask).sum() / mask.sum().clamp_min(1)

    def backward(self, loss, modules, case):
        loss.backward()

    def zero_grad(self, modules, case):
        modules.grad = None

    def canonicalize_for_check(self, output, prepared_batch):
        return output.detach()

    def communication_counters(self):
        return {"payload_bytes": 0}

    def close(self):
        self.closed = True


class GoldenMockAdapter(MockAdapter):
    def capture_golden(self, modules, prepared_batch, case):
        return {
            "loss": 1.25,
            "output_sha256": "output-hash",
            "_tensors": {"output": torch.arange(4)},
        }


def _small_case(name: str):
    payload = load_case(CASES / name).to_dict()
    for derived in ("selection", "world_size", "config_hash"):
        payload.pop(derived)
    payload["sequence"] = {"layout": "sbhd", "logical_lengths": [32], "padded_lengths": [32]}
    payload["parallel"] = {"tp": 1, "cp": 1, "pp": 1, "dp": 1}
    payload["warmup"] = 1
    payload["iterations"] = 2
    if "context_parallel" in payload["backend"]["capabilities"]:
        payload["backend"]["capabilities"].remove("context_parallel")
    if "packed_varlen" in payload["backend"]["capabilities"]:
        payload["backend"]["capabilities"].remove("packed_varlen")
    from exps.megatron_attention.config import AttentionBenchmarkCase

    return AttentionBenchmarkCase.from_dict(payload)


@pytest.mark.parametrize("case_name", ["smoke_mla_cp1.json", "hybrid_3to1_cp1.json"])
def test_one_runner_covers_both_harnesses(tmp_path: Path, case_name: str) -> None:
    case = _small_case(case_name)
    adapter = MockAdapter()
    output = tmp_path / f"{case.case_name}.json"
    result = BenchmarkRunner(case, adapter, output_path=output, repo_root=ROOT).run()
    assert result["status"] == "pass"
    assert result["completed_phases"] == list(BenchmarkRunner.PHASES)
    assert len(result["rank_local_samples_ms"]) == 2
    assert set(result["rank_local_samples_ms"][0]) == {
        "batch_materialize",
        "thd_pack",
        "layout_plan",
        "dispatch",
        "forward",
        "local_loss",
        "backward",
        "optimizer_zero",
        "headline_e2e",
    }
    assert json.loads(output.read_text())["status"] == "pass"
    assert adapter.closed


def test_failure_is_atomically_recorded_with_completed_phases(tmp_path: Path) -> None:
    case = _small_case("smoke_mla_cp1.json")
    adapter = MockAdapter(fail_forward=True)
    output = tmp_path / "failure.json"
    with pytest.raises(RuntimeError, match="intentional mock failure"):
        BenchmarkRunner(case, adapter, output_path=output, repo_root=ROOT).run()
    result = json.loads(output.read_text())
    assert result["status"] == "fail"
    assert result["error_type"] == "RuntimeError"
    assert result["completed_phases"] == ["validate", "init_groups", "materialize", "build"]
    assert "intentional mock failure" in result["traceback"]
    assert adapter.closed


def test_runner_persists_optional_golden_artifact(tmp_path: Path) -> None:
    case = _small_case("smoke_mla_cp1.json")
    output = tmp_path / "golden.json"

    result = BenchmarkRunner(
        case,
        GoldenMockAdapter(),
        output_path=output,
        repo_root=ROOT,
    ).run()

    golden = result["correctness"]["golden"]
    artifact = tmp_path / "golden.golden.pt"
    assert golden["loss"] == 1.25
    assert golden["artifact"] == str(artifact)
    assert torch.equal(torch.load(artifact)["output"], torch.arange(4))


def test_profiler_range_wraps_only_measured_iterations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _small_case("smoke_mla_cp1.json")
    events = []
    monkeypatch.setenv("MAGI_BENCH_CUDA_PROFILER_RANGE", "1")
    monkeypatch.setattr(runner_module, "_cuda_profiler_start", lambda: events.append("start"))
    monkeypatch.setattr(runner_module, "_cuda_profiler_stop", lambda: events.append("stop"))

    BenchmarkRunner(
        case,
        MockAdapter(),
        output_path=tmp_path / "profiled.json",
        repo_root=ROOT,
    ).run()

    assert events == ["start", "stop"]


def test_distributed_barrier_only_runs_for_initialized_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda: "nccl")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(
        torch.distributed,
        "barrier",
        lambda **kwargs: events.append(("barrier", kwargs)),
    )

    runner_module._distributed_barrier()
    assert events == []

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    runner_module._distributed_barrier()
    assert events == [("barrier", {"device_ids": [2]})]
