from __future__ import annotations

import argparse
import json
from pathlib import Path

from optimus.core.perturbations import perturbation_panel, write_perturbation_file


def parse_float_list(text: str) -> list[float]:
    return [float(item) for item in text.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write a deterministic zeroth-order perturbation panel.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--method", choices=["dense", "lora"], required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--population", type=int, required=True)
    parser.add_argument("--sigma", type=float, required=True)
    parser.add_argument("--sigma-values", default="")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--antithetic", action="store_true")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--targets", default="")
    parser.add_argument("--summary-out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sigma_values = parse_float_list(args.sigma_values) if args.sigma_values else None
    perturbations = perturbation_panel(
        args.method,
        args.family,
        args.population,
        args.sigma,
        args.seed,
        args.antithetic,
        sigma_values,
        rank=args.rank,
        targets=args.targets,
    )
    write_perturbation_file(args.out, perturbations)
    summary = {
        "kind": "perturbation_panel",
        "out": str(args.out),
        "method": args.method,
        "family": args.family,
        "population": len(perturbations),
        "sigma": args.sigma,
        "sigma_values": sigma_values or [args.sigma],
        "seed": args.seed,
        "antithetic": args.antithetic,
        "rank": args.rank,
        "targets": [item for item in args.targets.split(",") if item.strip()],
        "first": perturbations[0].to_record() if perturbations else None,
    }
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
