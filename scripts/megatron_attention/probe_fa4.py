"""Report FA4/CuTeDSL module and distribution provenance as JSON."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json


def main() -> None:
    distributions = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name", "")
        if "flash" in name.lower() or "cutlass" in name.lower():
            distributions[name] = distribution.version
    modules = {}
    for name in ("flash_attn", "flash_attn_cute", "flash_attn_cute.interface", "cutlass"):
        spec = importlib.util.find_spec(name)
        modules[name] = None if spec is None else spec.origin
    print(json.dumps({"distributions": distributions, "modules": modules}, indent=2))


if __name__ == "__main__":
    main()
