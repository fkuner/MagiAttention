"""Reproducibility metadata captured for every pass or failure result."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path


def collect_provenance(repo_root: Path) -> dict[str, object]:
    return {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "repo_root": str(repo_root.resolve()),
        "git": _git_state(repo_root),
        "environment": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "NCCL_DEBUG",
                "NCCL_ALGO",
                "NCCL_PROTO",
                "TORCH_NCCL_ASYNC_ERROR_HANDLING",
            )
        },
    }


def _git_state(repo_root: Path) -> dict[str, object]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run("status", "--short")
    return {
        "head": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status_short": status.splitlines() if status else [],
    }
