from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from optimus.runs.gpu_suite import GpuSuiteConfig, gpu_suite_specs, parse_int_tuple


@dataclass(frozen=True)
class RunContract:
    name: str
    root: Path
    required_files: tuple[str, ...]


@dataclass(frozen=True)
class RunCheck:
    name: str
    root: str
    required: int
    present: int
    missing: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.missing


def check_run(contract: RunContract) -> RunCheck:
    missing = tuple(rel for rel in contract.required_files if not (contract.root / rel).exists())
    return RunCheck(
        name=contract.name,
        root=str(contract.root),
        required=len(contract.required_files),
        present=len(contract.required_files) - len(missing),
        missing=missing,
    )


def gpu_suite_contracts(config: GpuSuiteConfig) -> list[RunContract]:
    contracts = []
    for spec in gpu_suite_specs(config):
        if spec.kind == "bench":
            required = ("summary.json", "adapter_rows.jsonl", "per_prompt.jsonl")
        elif spec.kind == "search":
            required = ("summary.json", "candidate_summary.jsonl", "per_prompt.jsonl", "holdout_per_prompt.jsonl")
        elif spec.kind == "halving":
            required = (
                "summary.json",
                "stage_candidate_summary.jsonl",
                "candidate_summary.jsonl",
                "stage_per_prompt.jsonl",
                "holdout_per_prompt.jsonl",
            )
        else:
            continue
        contracts.append(RunContract(spec.name, spec.output_path, required))
    contracts.append(
        RunContract(
            "systems_report",
            config.systems_output_root,
            (
                "report.md",
                "bench.csv",
                "adapter_throughput.png",
                "full_search.csv",
                "full_search_candidate_sec.png",
                "best_of_n.csv",
                "best_of_n.png",
                "quality_scaling.csv",
                "quality_scaling.png",
                "token_throughput.png",
                "parity.csv",
                "halving.csv",
            ),
        )
    )
    return contracts


def summary_payload(checks: list[RunCheck]) -> dict:
    return {
        "pass": all(check.passed for check in checks),
        "checks": [asdict(check) | {"pass": check.passed} for check in checks],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate an Optimus GPU run directory.")
    parser.add_argument("--root", type=Path, default=Path("results/optimus_gpu_suite"))
    parser.add_argument("--systems-out", type=Path, default=Path("results/report/optimus_systems"))
    parser.add_argument("--populations", default="1024,4096")
    parser.add_argument("--bench-adapters", default="8,16,32")
    parser.add_argument("--skip-halving", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = GpuSuiteConfig(
        output_root=args.root,
        systems_output_root=args.systems_out,
        populations=parse_int_tuple(args.populations),
        bench_adapters=parse_int_tuple(args.bench_adapters),
        run_halving=not args.skip_halving,
    )
    checks = [check_run(contract) for contract in gpu_suite_contracts(config)]
    payload = summary_payload(checks)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    else:
        print(text, end="")
    return 1 if args.strict and not payload["pass"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
