# T01 PR #331 preflight

Status: **static preflight passed; runtime Gate remains T02**.

## Provenance

- Base: `529fb0a4e273b3557a56d8afd60b74da46688095`
- PR head: `99f836e852990579363a366bf1276d828c3e53d4`
- Stable patch-id: `94b5f5be722bafc155b6d3805231a17e41cabf45`
- Remote worktree: `/root/work/MagiAttention-pr331-preflight`
- Worktree state: clean, detached at the exact PR head
- FlashAttention submodule: `43962bad5b45169b04130290cd62859f4b0c82e5`

## Safety result

The PR changes 17 paths. Its changed-path set has an empty intersection with all active `magic` dirty files captured by T00. The PR is not applied to the active local or remote `magic` checkout.

The PR changes `.gitmodules` and the FlashAttention submodule pointer. That submodule is excluded from Unison and remains managed only inside the isolated worktree until merge approval, so the active remote build products are not overwritten.

## Static checks

- `git diff --check 529fb0a..99f836e`: passed.
- Python `compileall` over the PR package and `tests/test_pipeline.py`: passed.
- Compile warnings came only from vendored CUTLASS/FlashAttention example code (`is` with string literals and one invalid escape sequence). No changed Magi Python file failed compilation.

## Runtime evidence carried into T02

- Project environment: `/root/work/MagiAttention/.venv`, Python 3.12.13.
- Magi core/communication extensions and FA4 helpers were previously built/imported in this environment.
- CP2 BF16 `Q/K=192,V=128` forward executed in the isolated PR worktree.
- Backward has not passed yet: the first attempt remained in FA4/CuTeDSL cold JIT and was gracefully interrupted. T02 must distinguish compile latency from a genuine compile/runtime failure and must not report backward as passed until an explicit completion marker is captured.

## Decision

There is no source-overlap or static-replay blocker. Continue to T02. Do not merge PR #331 into active `magic` until its required runtime tests pass and the user dirty hashes are rechecked.
