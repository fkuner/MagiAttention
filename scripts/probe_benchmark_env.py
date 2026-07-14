#!/usr/bin/env python3
"""Emit a machine-readable benchmark environment manifest."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _import_probe(name: str) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        module = importlib.import_module(name)
        return {
            "ok": True,
            "path": getattr(module, "__file__", None),
            "seconds": time.perf_counter() - started,
        }
    except Exception as exc:  # noqa: BLE001 - the manifest must retain import failures.
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "seconds": time.perf_counter() - started,
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--magi-root", type=Path, required=True)
    parser.add_argument("--megatron-root", type=Path, required=True)
    parser.add_argument("--gpu-model-override")
    parser.add_argument("--require-gpu", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    magi_root = args.magi_root.resolve()
    megatron_root = args.megatron_root.resolve()

    # Keep the checkout under test ahead of any editable/site-packages install.
    sys.path.insert(0, str(megatron_root))
    sys.path.insert(0, str(magi_root))

    import torch

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "platform": platform.platform(),
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
        },
        "paths": {
            "magi_root": str(magi_root),
            "megatron_root": str(megatron_root),
            "cwd": os.getcwd(),
        },
        "packages": {
            name: _distribution_version(name)
            for name in (
                "torch",
                "transformer-engine",
                "transformer-engine-torch",
                "nvidia-cutlass-dsl",
                "nvidia-cudnn-frontend",
                "nvidia-cublas",
                "nvidia-nccl-cu13",
            )
        },
        "torch": {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cudnn_version": torch.backends.cudnn.version(),
        },
    }

    try:
        manifest["torch"]["nccl_version"] = torch.cuda.nccl.version()
    except Exception as exc:  # noqa: BLE001
        manifest["torch"]["nccl_version_error"] = f"{type(exc).__name__}: {exc}"

    imports = {
        name: _import_probe(name)
        for name in (
            "magi_attention",
            "magi_attention.functional.fa4",
            "magi_attention.magi_attn_comm.grpcoll",
            "transformer_engine.pytorch",
            "megatron.core",
            "cutlass",
            "cutlass.cute",
        )
    }
    manifest["imports"] = imports

    magi_module_path = imports["magi_attention"].get("path")
    manifest["source_checkout_verified"] = bool(
        magi_module_path
        and Path(magi_module_path).resolve().is_relative_to(magi_root)
    )

    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    manifest["gpus"] = []
    for index in range(gpu_count):
        properties = torch.cuda.get_device_properties(index)
        manifest["gpus"].append(
            {
                "index": index,
                "name_raw": properties.name,
                "compute_capability": [properties.major, properties.minor],
                "total_memory_bytes": properties.total_memory,
                "multi_processor_count": properties.multi_processor_count,
            }
        )

    manifest["gpu_identity"] = {
        "driver_reported_names": sorted({gpu["name_raw"] for gpu in manifest["gpus"]}),
        "user_confirmed_model": args.gpu_model_override,
        "runtime_architecture": "SM100"
        if manifest["gpus"]
        and all(gpu["compute_capability"] == [10, 0] for gpu in manifest["gpus"])
        else None,
        "note": (
            "The driver name is retained verbatim. The model override is an explicit "
            "user-provided correction and is not inferred from the driver string."
        ),
    }

    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
        torch.manual_seed(1234)
        started = time.perf_counter()
        left = torch.randn((1024, 1024), device=device, dtype=torch.bfloat16)
        right = torch.randn((1024, 1024), device=device, dtype=torch.bfloat16)
        output = left @ right
        torch.cuda.synchronize(device)
        manifest["bf16_matmul"] = {
            "ok": bool(torch.isfinite(output).all().item()),
            "shape": list(output.shape),
            "dtype": str(output.dtype),
            "seconds_including_first_use": time.perf_counter() - started,
        }
    else:
        manifest["bf16_matmul"] = {"ok": False, "error": "CUDA is unavailable"}

    required_imports = (
        "magi_attention",
        "magi_attention.functional.fa4",
        "magi_attention.magi_attn_comm.grpcoll",
        "transformer_engine.pytorch",
        "megatron.core",
    )
    ok = all(imports[name]["ok"] for name in required_imports)
    ok = ok and manifest["source_checkout_verified"]
    if args.require_gpu:
        ok = ok and gpu_count > 0 and manifest["bf16_matmul"]["ok"]
    manifest["ok"] = ok
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
