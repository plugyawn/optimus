from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="optimus bench",
        description="Measure Optimus backend throughput through an explicit backend and method.",
    )
    parser.add_argument("--backend", required=True, choices=["vllm"])
    parser.add_argument("--method", required=True, choices=["lora", "subspace"])
    common = parser.add_argument_group("common bench options")
    common.add_argument("--out")
    common.add_argument("--model")
    common.add_argument("--data")
    common.add_argument("--tensor-parallel-size", type=int)
    common.add_argument("--max-new-tokens", type=int)
    common.add_argument("--prompt-input", choices=["text", "token_ids"])
    common.add_argument("--stop-at-answer", action="store_true")
    common.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=None)
    common.add_argument("--enable-chunked-prefill", action=argparse.BooleanOptionalAction, default=None)
    common.add_argument("--kv-cache-dtype")
    common.add_argument("--vllm-kwarg", action="append")
    lora = parser.add_argument_group("legacy LoRA bench options")
    lora.add_argument("--adapters", type=int, help="Number of LoRA adapters for explicit legacy adapter throughput baselines.")
    lora.add_argument("--prompts", type=int, help="Prompt count for throughput measurement.")
    lora.add_argument("--rank", type=int, help="LoRA rank for explicit legacy LoRA baselines.")
    lora.add_argument("--sigma", type=float, help="LoRA or dense perturbation scale for explicit legacy baselines.")
    lora.add_argument("--targets", help="Comma-separated LoRA target modules for explicit legacy baselines.")
    lora.add_argument("--max-loras", type=int, help="Maximum active LoRA adapters for explicit legacy LoRA baselines.")
    lora.add_argument("--max-cpu-loras", type=int, help="CPU LoRA cache size for explicit legacy LoRA baselines.")
    lora.add_argument("--preload", action="store_true", help="Preload LoRA adapters for explicit legacy baselines.")
    lora.add_argument("--mixed-batch", action="store_true", help="Run mixed-adapter batch for explicit legacy baselines.")
    lora.add_argument("--skip-sequential", action="store_true", help="Skip sequential adapter loop for explicit legacy baselines.")
    lora.add_argument("--no-include-base", action="store_true", help="Exclude base requests for explicit legacy baselines.")
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
    subspace.add_argument("--kernel", choices=["torch", "triton", "flashinfer", "custom-op"], help="Lazy delta kernel backend.")
    subspace.add_argument(
        "--prefix-cache-policy",
        choices=["disabled-for-search"],
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
    ns = build_parser().parse_args(args)
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
