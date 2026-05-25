from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="optimus search",
        description="Run Optimus perturbation search through an explicit backend and method.",
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

    if ns.backend == "transformers":
        if ns.method == "subspace":
            raise SystemExit(
                "optimus search --backend transformers --method subspace is reserved for the "
                "Phase 3 reference evaluator and is not implemented yet."
            )
        from optimus.search.peft import main as transformers_main

        return int(
            transformers_main(
                [
                    "search",
                    *passthrough,
                    "--perturbation-backend",
                    "dense" if ns.method == "dense" else "lora",
                ]
            )
            or 0
        )

    if ns.backend == "vllm" and ns.method == "lora":
        from optimus.serving.search import main as vllm_lora_main

        return int(vllm_lora_main(passthrough) or 0)

    if ns.backend == "vllm" and ns.method == "subspace":
        raise SystemExit(
            "optimus search --backend vllm --method subspace is the planned production "
            "path, but the vLLM subspace backend is not implemented yet. See "
            "docs/subspace_implementation_roadmap.md Phase 5."
        )

    raise SystemExit(f"unsupported search route: backend={ns.backend!r}, method={ns.method!r}")


if __name__ == "__main__":
    raise SystemExit(main())
