from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

FORBIDDEN_PUBLIC_PASSTHROUGH = {
    "--activation-state-prompts",
    "--activation-state-batch-size",
    "--activation-state-no-anchor-subtract",
    "--family-state-file",
    "--stage-prompts",
    "--survivors",
    "--batch-sizes",
    "--prompt-counts",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="optimus search",
        description="Run Optimus perturbation search through an explicit backend and method.",
    )
    parser.add_argument("--backend", required=True, choices=["transformers", "vllm"])
    parser.add_argument("--method", required=True, choices=["dense", "lora", "subspace"])
    lora = parser.add_argument_group("legacy LoRA search options")
    lora.add_argument("--rank", type=int, help="LoRA rank for explicit legacy LoRA baselines.")
    lora.add_argument("--sigma", type=float, help="LoRA or dense perturbation scale for legacy baselines.")
    lora.add_argument("--targets", help="Comma-separated LoRA target modules for explicit legacy baselines.")
    subspace = parser.add_argument_group("subspace search options")
    subspace.add_argument("--basis-rank", type=int, help="Activation-site basis rank.")
    subspace.add_argument("--basis-prompts", type=int, help="Number of prompts used for basis calibration.")
    subspace.add_argument("--target-preset", choices=["qv", "attn-qkvo", "mlp", "transformer-linears"])
    subspace.add_argument("--layers", default=None, help="Layer selector, for example 'all' or '0,1,2'.")
    subspace.add_argument("--basis-centering", choices=["none", "mean"])
    subspace.add_argument("--basis-token-source", choices=["prefill", "decode", "prefill+decode"])
    subspace.add_argument("--basis-kind", choices=["activation-svd", "random-orthonormal", "shuffled-activation-svd"])
    subspace.add_argument("--scale-mode", choices=["projected-dense", "relative-output-rms"])
    subspace.add_argument("--rho-grid", help="Comma-separated relative output RMS radii.")
    subspace.add_argument("--sigma-w-grid", help="Comma-separated dense weight-noise scales.")
    subspace.add_argument(
        "--budget-policy",
        choices=["raw-dense", "per-target-equal", "per-layer-equal", "per-block-equal", "custom-json"],
    )
    subspace.add_argument("--top-k-grid", help="Comma-separated lazy ensemble K values.")
    subspace.add_argument("--candidate-batch-size", help="Candidate block size or 'auto'.")
    subspace.add_argument("--kernel", choices=["torch", "triton", "custom-op"], help="Lazy delta kernel backend.")
    subspace.add_argument(
        "--prefix-cache-policy",
        choices=["disabled-for-search", "candidate-keyed"],
        help="Prefix-cache behavior for candidate-specific KV state.",
    )
    subspace.add_argument("--match-screen-to-holdout-base-exact", action="store_true")
    subspace.add_argument("--screen-pool-prompts", type=int)
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


def _reject_forbidden_passthrough(argv: Sequence[str]) -> None:
    for item in argv:
        option = item.split("=", 1)[0]
        if option in FORBIDDEN_PUBLIC_PASSTHROUGH:
            raise SystemExit(f"{option} is not part of the Optimus public search surface")


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    _reject_forbidden_passthrough(args)
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
