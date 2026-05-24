from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path

from optimus.runs.gpu_suite import GpuSuiteConfig, gpu_suite_specs, parse_int_tuple


@dataclass(frozen=True)
class RunContract:
    name: str
    root: Path
    required_files: tuple[str, ...]
    required_summary_keys: tuple[str, ...] = ()
    required_positive_keys: tuple[str, ...] = ()
    required_finite_keys: tuple[str, ...] = ()
    required_nonempty_keys: tuple[str, ...] = ()
    required_path_keys: tuple[str, ...] = ()
    expected_summary_values: dict[str, object] | None = None
    required_jsonl_nonempty: tuple[str, ...] = ()
    required_jsonl_fields: dict[str, tuple[str, ...]] | None = None
    expected_jsonl_counts: dict[str, str] | None = None
    required_csv_nonempty: tuple[str, ...] = ()
    required_bool_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class RunCheck:
    name: str
    root: str
    required: int
    present: int
    missing: tuple[str, ...]
    invalid: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return not self.missing and not self.invalid


def check_run(contract: RunContract) -> RunCheck:
    missing = tuple(rel for rel in contract.required_files if not (contract.root / rel).exists())
    invalid: list[str] = []
    summary_path = contract.root / "summary.json"
    summary = {}
    if "summary.json" in contract.required_files and summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except json.JSONDecodeError:
            invalid.append("summary.json: invalid JSON")
    for key in contract.required_summary_keys:
        if key not in summary:
            invalid.append(f"summary.{key}: missing")
    for key, expected in (contract.expected_summary_values or {}).items():
        observed = summary.get(key)
        if observed != expected:
            invalid.append(f"summary.{key}: expected {expected!r}, got {observed!r}")
    for key in contract.required_positive_keys:
        value = summary.get(key)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            invalid.append(f"summary.{key}: not numeric")
            continue
        if not isfinite(numeric) or numeric <= 0:
            invalid.append(f"summary.{key}: not positive")
    for key in contract.required_finite_keys:
        value = summary.get(key)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            invalid.append(f"summary.{key}: not numeric")
            continue
        if not isfinite(numeric):
            invalid.append(f"summary.{key}: not finite")
    for key in contract.required_bool_keys:
        if not isinstance(summary.get(key), bool):
            invalid.append(f"summary.{key}: not boolean")
    for key in contract.required_nonempty_keys:
        value = summary.get(key)
        if not value:
            invalid.append(f"summary.{key}: empty")
    jsonl_counts: dict[str, int] = {}
    for rel in contract.required_jsonl_nonempty:
        path = contract.root / rel
        if not path.exists():
            continue
        try:
            count = 0
            with path.open() as f:
                for line_no, line in enumerate(f, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    for field in (contract.required_jsonl_fields or {}).get(rel, ()):
                        if field not in row:
                            invalid.append(f"{rel}: row {line_no} missing {field}")
                    count += 1
            jsonl_counts[rel] = count
            if count == 0:
                invalid.append(f"{rel}: empty")
        except json.JSONDecodeError as exc:
            invalid.append(f"{rel}: invalid JSONL at line {line_no}: {exc.msg}")
    for rel, summary_key in (contract.expected_jsonl_counts or {}).items():
        if rel not in jsonl_counts or summary_key not in summary:
            continue
        expected = int(summary[summary_key])
        if jsonl_counts[rel] != expected:
            invalid.append(f"{rel}: rows {jsonl_counts[rel]} != summary.{summary_key} {expected}")
    for rel in contract.required_csv_nonempty:
        path = contract.root / rel
        if not path.exists():
            continue
        try:
            with path.open(newline="") as f:
                rows = list(csv.DictReader(f))
        except csv.Error as exc:
            invalid.append(f"{rel}: invalid CSV: {exc}")
            continue
        if not rows:
            invalid.append(f"{rel}: no data rows")
    for rel in contract.required_files:
        path = contract.root / rel
        if path.exists() and path.suffix not in {".json", ".jsonl", ".csv"}:
            if path.stat().st_size <= 0:
                invalid.append(f"{rel}: empty file")
            if path.suffix == ".png":
                with path.open("rb") as f:
                    if f.read(8) != b"\x89PNG\r\n\x1a\n":
                        invalid.append(f"{rel}: invalid PNG signature")
    for key in contract.required_path_keys:
        value = summary.get(key)
        if not value:
            invalid.append(f"summary.{key}: missing")
            continue
        path = Path(str(value))
        if not path.exists():
            invalid.append(f"summary.{key}: path does not exist: {value}")
    return RunCheck(
        name=contract.name,
        root=str(contract.root),
        required=len(contract.required_files),
        present=len(contract.required_files) - len(missing),
        missing=missing,
        invalid=tuple(invalid),
    )


def gpu_suite_contracts(config: GpuSuiteConfig) -> list[RunContract]:
    contracts = []
    for spec in gpu_suite_specs(config):
        if spec.kind == "bench":
            required = ("summary.json", "adapter_rows.jsonl", "per_prompt.jsonl")
            required_summary = ("kind", "method", "adapter_build_s", "load_s", "lora_tokens_per_sec", "mixed_tokens_per_sec", "mixed_prompts_per_sec")
            required_positive = ("adapter_build_s", "load_s", "mixed_tokens_per_sec", "mixed_prompts_per_sec")
            required_finite = ()
            required_nonempty = ()
            required_path_keys = ()
            expected_summary_values = {"kind": "vllm_lora_bench", "method": "lora"}
            required_jsonl_nonempty = ("adapter_rows.jsonl", "per_prompt.jsonl")
            required_jsonl_fields = {"adapter_rows.jsonl": ("mode",), "per_prompt.jsonl": ("mode", "candidate")}
            expected_jsonl_counts = {}
            required_bool_keys = ()
        elif spec.kind == "search":
            required = ("summary.json", "candidate_summary.jsonl", "per_prompt.jsonl", "holdout_per_prompt.jsonl")
            required_summary = (
                "kind",
                "method",
                "population",
                "base_holdout_exact",
                "candidate_sec",
                "screen_prompts_per_sec",
                "screen_tokens_per_sec",
                "holdout_tokens_per_sec",
                "best_tokens_per_sec",
                "eval_elapsed_s",
                "load_s",
                "top_screen",
                "top_holdout",
            )
            required_positive = (
                "population",
                "candidate_sec",
                "screen_prompts_per_sec",
                "screen_tokens_per_sec",
                "holdout_tokens_per_sec",
                "best_tokens_per_sec",
                "eval_elapsed_s",
            )
            required_finite = ()
            required_nonempty = ("top_screen", "top_holdout")
            required_path_keys = ()
            expected_summary_values = {"kind": "vllm_lora_search", "method": "lora"}
            required_jsonl_nonempty = ("candidate_summary.jsonl", "per_prompt.jsonl", "holdout_per_prompt.jsonl")
            required_jsonl_fields = {
                "candidate_summary.jsonl": ("candidate", "exact_mean"),
                "per_prompt.jsonl": ("mode", "candidate"),
                "holdout_per_prompt.jsonl": ("mode", "candidate"),
            }
            expected_jsonl_counts = {"candidate_summary.jsonl": "population"}
            required_bool_keys = ()
        elif spec.kind == "halving":
            required = (
                "summary.json",
                "stage_candidate_summary.jsonl",
                "candidate_summary.jsonl",
                "stage_per_prompt.jsonl",
                "holdout_per_prompt.jsonl",
            )
            required_summary = (
                "kind",
                "method",
                "population",
                "base_holdout_exact",
                "candidate_sec",
                "stage_candidate_sec",
                "screen_candidate_sec",
                "prompt_eval_savings",
                "eval_elapsed_s",
                "top_stage",
                "top_screen",
                "top_holdout",
                "top8_survivor_recall",
                "top8_possible",
                "full_best_survived",
                "halving_selected_regret_vs_full",
            )
            required_positive = ("population", "candidate_sec", "stage_candidate_sec", "screen_candidate_sec", "eval_elapsed_s")
            required_finite = ("top8_survivor_recall", "halving_selected_regret_vs_full")
            required_nonempty = ("top_stage", "top_screen")
            required_path_keys = ("full_search_reference",)
            expected_summary_values = {"kind": "vllm_lora_halving", "method": "lora"}
            required_jsonl_nonempty = (
                "stage_candidate_summary.jsonl",
                "candidate_summary.jsonl",
                "stage_per_prompt.jsonl",
                "holdout_per_prompt.jsonl",
            )
            required_jsonl_fields = {
                "stage_candidate_summary.jsonl": ("candidate", "exact_mean"),
                "candidate_summary.jsonl": ("candidate", "exact_mean"),
                "stage_per_prompt.jsonl": ("mode", "candidate"),
                "holdout_per_prompt.jsonl": ("mode", "candidate"),
            }
            expected_jsonl_counts = {"stage_candidate_summary.jsonl": "population", "candidate_summary.jsonl": "survivors"}
            required_bool_keys = ("full_best_survived",)
        else:
            continue
        contracts.append(
            RunContract(
                spec.name,
                spec.output_path,
                required,
                required_summary,
                required_positive,
                required_finite,
                required_nonempty,
                required_path_keys,
                expected_summary_values,
                required_jsonl_nonempty,
                required_jsonl_fields,
                expected_jsonl_counts,
                required_bool_keys=required_bool_keys,
            )
        )
    systems_csvs = ["bench.csv", "full_search.csv", "best_of_n.csv", "quality_scaling.csv", "parity.csv"]
    systems_files = [
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
    ]
    if config.run_halving:
        systems_csvs.append("halving.csv")
        systems_files.append("halving_tradeoff.png")
    contracts.append(
        RunContract(
            "systems_report",
            config.systems_output_root,
            tuple(systems_files),
            required_csv_nonempty=tuple(systems_csvs),
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
