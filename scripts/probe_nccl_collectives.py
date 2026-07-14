#!/usr/bin/env python3
"""Validate the NCCL process group with deterministic GPU collectives."""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist


def main() -> int:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)

    reduced = torch.tensor([rank + 1.0], device=device, dtype=torch.float32)
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    expected_sum = world_size * (world_size + 1) / 2

    local = torch.tensor([rank], device=device, dtype=torch.int64)
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    gathered_values = [int(item.item()) for item in gathered]

    ok = reduced.item() == expected_sum and gathered_values == list(range(world_size))
    result = {
        "backend": dist.get_backend(),
        "world_size": world_size,
        "all_reduce": float(reduced.item()),
        "all_reduce_expected": expected_sum,
        "all_gather": gathered_values,
        "ok": ok,
    }
    dist.barrier()
    if rank == 0:
        print(json.dumps(result, sort_keys=True))
    dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
