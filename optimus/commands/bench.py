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
    lora = parser.add_argument_group("legacy LoRA bench options")
    lora.add_argument("--adapters", type=int, help="Number of LoRA adapters for explicit legacy adapter throughput baselines.")
    lora.add_argument("--prompts", type=int, help="Prompt count for throughput measurement.")
    subspace = parser.add_argument_group("subspace bench options")
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
