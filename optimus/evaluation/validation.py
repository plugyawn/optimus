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
    required_json_fields: dict[str, tuple[str, ...]] | None = None
    required_json_nonempty_fields: dict[str, tuple[str, ...]] | None = None


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
    for rel, fields in (contract.required_json_fields or {}).items():
        path = contract.root / rel
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            invalid.append(f"{rel}: invalid JSON")
            continue
        for field in fields:
            if field not in payload:
                invalid.append(f"{rel}.{field}: missing")
        for field in (contract.required_json_nonempty_fields or {}).get(rel, ()):
            value = payload.get(field)
            if not value:
                invalid.append(f"{rel}.{field}: empty")
        if rel == "top_k_ensemble.json" and "candidates" in fields:
            candidates = payload.get("candidates")
            if isinstance(candidates, list) and candidates:
                candidate_required = (
                    "candidate_id",
                    "direction_seed",
                    "sign",
                    "basis_hash",
                    "target_set_hash",
                    "scale_mode",
                    "budget_policy",
                    "rng_version",
                    "runtime_dtype",
                )
                for index, candidate in enumerate(candidates, start=1):
                    if not isinstance(candidate, dict):
                        invalid.append(f"{rel}.candidates[{index}]: not object")
                        continue
                    for field in candidate_required:
                        if field not in candidate:
                            invalid.append(f"{rel}.candidates[{index}].{field}: missing")
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
            if spec.method == "subspace":
                required = (
                    "summary.json",
                    "subspace_state.pt",
                    "subspace_state_summary.json",
                    "candidates.jsonl",
                    "candidate_scores.jsonl",
                    "top_k_ensemble.json",
                    "validation_report.json",
                    "systems_report.json",
                )
                required_summary = (
                    "kind",
                    "backend",
                    "method",
                    "population",
                    "basis_hash",
                    "target_set_hash",
                    "scale_mode",
                    "budget_policy",
                    "rng_version",
                    "candidate_routing",
                    "prefix_cache_policy",
                    "scorer_version",
                    "prompt_ids_hash",
                    "decode_config_hash",
                    "candidates_per_sec",
                    "prompts_per_sec",
                    "output_tokens_per_sec",
                    "lazy_overhead_pct",
                )
                required_positive = ("population", "candidates_per_sec", "prompts_per_sec", "output_tokens_per_sec")
                required_finite = ("lazy_overhead_pct",)
                required_nonempty = ("basis_hash", "target_set_hash", "scorer_version", "prompt_ids_hash", "decode_config_hash")
                required_path_keys = ()
                expected_summary_values = {"kind": f"subspace_{spec.backend}_search", "backend": spec.backend, "method": "subspace"}
                required_jsonl_nonempty = ("candidates.jsonl", "candidate_scores.jsonl")
                required_jsonl_fields = {
                    "candidates.jsonl": (
                        "candidate_id",
                        "direction_seed",
                        "sign",
                        "basis_hash",
                        "target_set_hash",
                        "scale_mode",
                        "budget_policy",
                        "rng_version",
                    ),
                    "candidate_scores.jsonl": ("candidate_id", "screen_score", "scorer_version", "prompt_ids_hash"),
                }
                expected_jsonl_counts = {"candidates.jsonl": "population"}
                required_bool_keys = ()
                required_json_fields = {
                    "subspace_state_summary.json": ("schema_version", "basis_hash", "activation_sites", "targets"),
                    "top_k_ensemble.json": (
                        "ensemble_kind",
                        "schema_version",
                        "aggregation",
                        "tie_break_policy",
                        "selection_rule",
                        "K",
                        "candidates",
                        "basis_hash",
                        "target_set_hash",
                        "scorer_version",
                        "prompt_ids_hash",
                        "runtime_config_hash",
                        "decode_config_hash",
                    ),
                    "validation_report.json": ("schema_version", "scientific_gate", "drift_diagnostics", "diversity_metrics"),
                    "systems_report.json": (
                        "schema_version",
                        "candidates_per_sec",
                        "prompts_per_sec",
                        "output_tokens_per_sec",
                        "lazy_overhead_pct",
                    ),
                }
                required_json_nonempty_fields = {
                    "subspace_state_summary.json": ("basis_hash", "activation_sites", "targets"),
                    "top_k_ensemble.json": ("candidates", "basis_hash", "target_set_hash", "scorer_version", "prompt_ids_hash"),
                }
            else:
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
                required_json_fields = {}
                required_json_nonempty_fields = {}
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
                required_json_fields=required_json_fields if spec.kind == "search" else None,
                required_json_nonempty_fields=required_json_nonempty_fields if spec.kind == "search" else None,
            )
        )
    if config.method == "subspace":
        systems_csvs = []
        systems_files = ["report.md", "systems_report.json"]
        systems_json_fields = {
            "systems_report.json": (
                "schema_version",
                "candidates_per_sec",
                "prompts_per_sec",
                "output_tokens_per_sec",
                "base_model_time_s",
                "qx_time_s",
                "lazy_delta_time_s",
                "scoring_time_s",
                "setup_time_s",
                "lazy_overhead_pct",
                "gpu_memory_allocated_bytes",
                "candidate_batch_size",
            )
        }
    else:
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
        ]
        systems_json_fields = {}
        if config.run_halving:
            systems_csvs.append("halving.csv")
            systems_files.append("halving_tradeoff.png")
    contracts.append(
        RunContract(
            "systems_report",
            config.systems_output_root,
            tuple(systems_files),
            required_csv_nonempty=tuple(systems_csvs),
            required_json_fields=systems_json_fields,
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
    parser.add_argument("--backend", choices=["vllm"], default="vllm")
    parser.add_argument("--method", choices=["lora", "subspace"], default="lora")
    parser.add_argument("--populations", default="1024,4096")
    parser.add_argument("--bench-adapters", default="8,16,32")
    parser.add_argument("--run-halving", action="store_true", help="Reserved until a final staged-search route exists.")
    parser.add_argument("--skip-halving", action="store_true", help="Compatibility no-op; staged search is disabled by default.")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = GpuSuiteConfig(
        output_root=args.root,
        systems_output_root=args.systems_out,
        backend=args.backend,
        method=args.method,
        populations=parse_int_tuple(args.populations),
        bench_adapters=parse_int_tuple(args.bench_adapters),
        run_halving=args.run_halving and not args.skip_halving,
    )
    try:
        contracts = gpu_suite_contracts(config)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None
    checks = [check_run(contract) for contract in contracts]
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
