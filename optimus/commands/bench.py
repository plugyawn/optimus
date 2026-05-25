from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="optimus bench",
        description="Measure Optimus backend throughput through an explicit backend and method.",
    )
    parser.add_argument("--backend", required=True, choices=["transformers", "vllm"])
    parser.add_argument("--method", required=True, choices=["dense", "lora", "subspace"])
    return parser


def _without_route_args(argv: Sequence[str]) -> list[str]:
    out: list[str] = []
    skip_next = False
    for idx, item in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if item in {"--backend", "--method"}:
            skip_next = idx + 1 < len(argv)
            continue
        if item.startswith("--backend=") or item.startswith("--method="):
            continue
        out.append(item)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    ns, _unknown = build_parser().parse_known_args(args)
    passthrough = _without_route_args(args)

    if ns.backend == "vllm" and ns.method == "lora":
        from optimus.serving.benchmark import main as vllm_lora_bench

        return int(vllm_lora_bench(passthrough) or 0)

    if ns.backend == "vllm" and ns.method == "subspace":
        raise SystemExit(
            "optimus bench --backend vllm --method subspace is reserved for the "
            "Phase 6 subspace speed gate and is not implemented yet."
        )

    raise SystemExit(f"unsupported bench route: backend={ns.backend!r}, method={ns.method!r}")


if __name__ == "__main__":
    raise SystemExit(main())
