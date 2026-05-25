from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path

from optimus.runs.gpu_suite import GpuSuiteConfig, gpu_suite_specs, parse_int_tuple


SUBSPACE_REQUIRED_FILES = (
    "summary.json",
    "subspace_state.pt",
    "subspace_state_summary.json",
    "candidates.jsonl",
    "candidate_scores.jsonl",
    "top_k_ensemble.json",
    "validation_report.json",
    "systems_report.json",
)
SUBSPACE_REQUIRED_SUMMARY = (
    "schema_version",
    "kind",
    "backend",
    "method",
    "created_at",
    "optimus_version",
    "git_commit",
    "git_dirty",
    "command",
    "environment",
    "population",
    "basis_hash",
    "target_set_hash",
    "basis_collection_config_hash",
    "subspace_state_hash",
    "candidate_scores_hash",
    "scale_mode",
    "rho_grid",
    "sigma_w_grid",
    "budget_policy",
    "rng_version",
    "candidate_routing",
    "prefix_cache_policy",
    "scorer_name",
    "scorer_version",
    "prompt_ids_hash",
    "sample_set_hash",
    "prompt_scoring_config_hash",
    "decode_config_hash",
    "candidates_per_sec",
    "prompts_per_sec",
    "output_tokens_per_sec",
    "lazy_overhead_pct",
)
SUBSPACE_CANDIDATE_FIELDS = (
    "candidate_id",
    "direction_seed",
    "sign",
    "basis_hash",
    "target_set_hash",
    "scale_mode",
    "rho_or_sigma_w",
    "budget_policy",
    "budget_hash",
    "rng_version",
    "runtime_dtype",
    "radius_index",
    "target_preset",
    "basis_rank",
    "shard_id",
    "shard_population_start",
    "shard_population_end",
    "worker_id",
    "device_id",
    "prompt_scoring_config_hash",
)
SUBSPACE_CANDIDATE_SCORE_FIELDS = (
    "candidate_id",
    "split",
    "scorer_name",
    "scorer_version",
    "aggregate_metrics",
    "sample_count",
    "prompt_ids_hash",
    "sample_set_hash",
    "decode_config_hash",
    "elapsed_s",
    "output_tokens",
)
SUBSPACE_TOP_K_FIELDS = (
    "ensemble_kind",
    "schema_version",
    "aggregation",
    "tie_break_policy",
    "selection_rule",
    "K",
    "candidates",
    "basis_hash",
    "basis_collection_config_hash",
    "subspace_state_hash",
    "scale_mode",
    "rho_or_sigma_w",
    "budget_policy",
    "target_set_hash",
    "candidate_scores_hash",
    "rng_version",
    "scorer_version",
    "prompt_ids_hash",
    "sample_set_hash",
    "prompt_scoring_config_hash",
    "runtime_config_hash",
    "decode_config_hash",
)
SUBSPACE_VALIDATION_SECTIONS = (
    "math_tests",
    "rng_replay_tests",
    "routing_cache_tests",
    "selector_quality",
    "holdout_quality",
    "ensemble_quality",
    "drift_diagnostics",
    "random_shuffled_controls",
    "throughput_gates",
)
SUBSPACE_SYSTEMS_FIELDS = (
    "schema_version",
    "warmup_policy",
    "cuda_sync_policy",
    "candidate_batch_size",
    "candidate_shard_id",
    "gpu_model",
    "gpu_count",
    "gpu_memory_allocated_bytes",
    "gpu_memory_reserved_bytes",
    "base_model_time_s",
    "qx_time_s",
    "lazy_delta_time_s",
    "scoring_time_s",
    "setup_time_s",
    "candidates_per_sec",
    "prompts_per_sec",
    "output_tokens_per_sec",
    "lazy_overhead_pct",
    "prefix_cache_policy",
    "top_k_ensemble_cost_multiplier",
    "screen_score",
    "holdout_score",
    "screen_to_holdout_drop",
    "diversity_metrics",
    "random_q_control",
    "shuffled_q_control",
    "antithetic_odd_even",
)


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
    expected_json_values: dict[str, dict[str, object]] | None = None


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


SUBSPACE_SCALE_MODES = {"relative-output-rms", "projected-dense"}
SUBSPACE_BUDGET_POLICIES = {"raw-dense", "per-target-equal", "per-layer-equal", "per-block-equal", "custom-json"}
SUBSPACE_SIGNS = {"+", "-"}
SUBSPACE_IDENTITY_FIELDS = (
    "basis_hash",
    "target_set_hash",
    "basis_collection_config_hash",
    "scale_mode",
    "budget_policy",
    "rng_version",
    "prompt_scoring_config_hash",
)
SUBSPACE_CANDIDATE_IDENTITY_FIELDS = (
    "basis_hash",
    "target_set_hash",
    "scale_mode",
    "budget_policy",
    "rng_version",
    "prompt_scoring_config_hash",
)


def _is_finite_number(value: object) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return isfinite(numeric)


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _json_identity(summary: dict, field: str) -> object:
    return summary.get(field)


def _check_identity_consistency(
    *,
    invalid: list[str],
    prefix: str,
    payload: dict,
    summary: dict,
    fields: tuple[str, ...] = SUBSPACE_IDENTITY_FIELDS,
) -> None:
    for field in fields:
        expected = _json_identity(summary, field)
        if expected is None:
            continue
        if payload.get(field) != expected:
            invalid.append(f"{prefix}.{field}: expected summary.{field} {expected!r}, got {payload.get(field)!r}")


def _check_subspace_candidate(invalid: list[str], prefix: str, row: dict, summary: dict) -> None:
    if row.get("sign") not in SUBSPACE_SIGNS:
        invalid.append(f"{prefix}.sign: invalid")
    if row.get("scale_mode") not in SUBSPACE_SCALE_MODES:
        invalid.append(f"{prefix}.scale_mode: invalid")
    if row.get("budget_policy") not in SUBSPACE_BUDGET_POLICIES:
        invalid.append(f"{prefix}.budget_policy: invalid")
    if not _is_finite_number(row.get("rho_or_sigma_w")):
        invalid.append(f"{prefix}.rho_or_sigma_w: not finite")
    for field in ("direction_seed", "radius_index", "shard_population_start", "shard_population_end"):
        if not _is_nonnegative_int(row.get(field)):
            invalid.append(f"{prefix}.{field}: not nonnegative integer")
    if not _is_positive_int(row.get("basis_rank")):
        invalid.append(f"{prefix}.basis_rank: not positive integer")
    start = row.get("shard_population_start")
    end = row.get("shard_population_end")
    if _is_nonnegative_int(start) and _is_nonnegative_int(end) and end <= start:
        invalid.append(f"{prefix}.shard_population_end: must exceed shard_population_start")
    _check_identity_consistency(
        invalid=invalid,
        prefix=prefix,
        payload=row,
        summary=summary,
        fields=SUBSPACE_CANDIDATE_IDENTITY_FIELDS,
    )


def _check_subspace_score(invalid: list[str], prefix: str, row: dict, summary: dict) -> None:
    if row.get("split") not in {"screen", "holdout", "validation", "test"}:
        invalid.append(f"{prefix}.split: invalid")
    if not isinstance(row.get("aggregate_metrics"), dict) or not row.get("aggregate_metrics"):
        invalid.append(f"{prefix}.aggregate_metrics: empty")
    if not _is_positive_int(row.get("sample_count")):
        invalid.append(f"{prefix}.sample_count: not positive integer")
    for field in ("elapsed_s", "output_tokens"):
        if not _is_finite_number(row.get(field)):
            invalid.append(f"{prefix}.{field}: not finite")
    for field in ("scorer_version", "prompt_ids_hash", "sample_set_hash", "decode_config_hash"):
        expected = summary.get(field)
        if expected is not None and row.get(field) != expected:
            invalid.append(f"{prefix}.{field}: expected summary.{field} {expected!r}, got {row.get(field)!r}")


def _evidence_path_exists(root: Path, value: str) -> bool:
    path = Path(value)
    return path.exists() if path.is_absolute() else (root / path).exists()


def check_run(contract: RunContract) -> RunCheck:
    missing = tuple(rel for rel in contract.required_files if not (contract.root / rel).exists())
    invalid: list[str] = []
    is_subspace_contract = "subspace_state.pt" in contract.required_files
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
    if is_subspace_contract:
        if not isinstance(summary.get("git_dirty"), bool):
            invalid.append("summary.git_dirty: not boolean")
        if not summary.get("command"):
            invalid.append("summary.command: empty")
        if not isinstance(summary.get("environment"), dict) or not summary.get("environment"):
            invalid.append("summary.environment: empty")
        scale_mode = summary.get("scale_mode")
        if scale_mode not in SUBSPACE_SCALE_MODES:
            invalid.append("summary.scale_mode: invalid")
        if scale_mode == "relative-output-rms":
            if not isinstance(summary.get("rho_grid"), list) or not summary.get("rho_grid"):
                invalid.append("summary.rho_grid: empty for relative-output-rms")
            elif not all(_is_finite_number(item) for item in summary["rho_grid"]):
                invalid.append("summary.rho_grid: contains non-finite value")
        if scale_mode == "projected-dense":
            if not isinstance(summary.get("sigma_w_grid"), list) or not summary.get("sigma_w_grid"):
                invalid.append("summary.sigma_w_grid: empty for projected-dense")
            elif not all(_is_finite_number(item) for item in summary["sigma_w_grid"]):
                invalid.append("summary.sigma_w_grid: contains non-finite value")
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
                    if is_subspace_contract and rel == "candidates.jsonl":
                        _check_subspace_candidate(invalid, f"{rel}: row {line_no}", row, summary)
                    elif rel == "candidates.jsonl":
                        if row.get("scale_mode") not in SUBSPACE_SCALE_MODES:
                            invalid.append(f"{rel}: row {line_no} invalid scale_mode")
                        if row.get("budget_policy") not in SUBSPACE_BUDGET_POLICIES:
                            invalid.append(f"{rel}: row {line_no} invalid budget_policy")
                    if is_subspace_contract and rel == "candidate_scores.jsonl":
                        _check_subspace_score(invalid, f"{rel}: row {line_no}", row, summary)
                    elif rel == "candidate_scores.jsonl":
                        if row.get("split") not in {"screen", "holdout", "validation", "test"}:
                            invalid.append(f"{rel}: row {line_no} invalid split")
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
        for field, expected in (contract.expected_json_values or {}).get(rel, {}).items():
            observed = payload.get(field)
            if observed != expected:
                invalid.append(f"{rel}.{field}: expected {expected!r}, got {observed!r}")
        if rel == "top_k_ensemble.json" and "candidates" in fields:
            _check_identity_consistency(invalid=invalid, prefix=rel, payload=payload, summary=summary)
            for field in ("subspace_state_hash", "candidate_scores_hash"):
                expected = summary.get(field)
                if expected is not None and payload.get(field) != expected:
                    invalid.append(f"{rel}.{field}: expected summary.{field} {expected!r}, got {payload.get(field)!r}")
            if payload.get("aggregation") not in {"majority-vote", "mean-logprob", "score-sum"}:
                invalid.append(f"{rel}.aggregation: invalid")
            if payload.get("tie_break_policy") not in {"lowest_candidate_id"}:
                invalid.append(f"{rel}.tie_break_policy: invalid")
            if not isinstance(payload.get("K"), int) or payload.get("K") <= 0:
                invalid.append(f"{rel}.K: not positive integer")
            if not _is_finite_number(payload.get("rho_or_sigma_w")):
                invalid.append(f"{rel}.rho_or_sigma_w: not finite")
            candidates = payload.get("candidates")
            if isinstance(candidates, list) and candidates:
                if payload.get("K") != len(candidates):
                    invalid.append(f"{rel}.K: {payload.get('K')!r} != len(candidates) {len(candidates)}")
                for index, candidate in enumerate(candidates, start=1):
                    if not isinstance(candidate, dict):
                        invalid.append(f"{rel}.candidates[{index}]: not object")
                        continue
                    for field in SUBSPACE_CANDIDATE_FIELDS:
                        if field not in candidate:
                            invalid.append(f"{rel}.candidates[{index}].{field}: missing")
                    _check_subspace_candidate(invalid, f"{rel}.candidates[{index}]", candidate, summary)
        if rel == "validation_report.json":
            required_sections = fields
            for section in required_sections:
                value = payload.get(section)
                if not isinstance(value, dict) or not value:
                    invalid.append(f"{rel}.{section}: empty")
                    continue
                for field in ("status", "evidence_paths", "failures"):
                    if field not in value:
                        invalid.append(f"{rel}.{section}.{field}: missing")
                if value.get("status") != "pass":
                    invalid.append(f"{rel}.{section}.status: expected 'pass', got {value.get('status')!r}")
                evidence_paths = value.get("evidence_paths")
                if not isinstance(evidence_paths, list) or not evidence_paths:
                    invalid.append(f"{rel}.{section}.evidence_paths: empty")
                else:
                    for evidence in evidence_paths:
                        if not isinstance(evidence, str) or not evidence or not _evidence_path_exists(contract.root, evidence):
                            invalid.append(f"{rel}.{section}.evidence_paths: missing {evidence!r}")
                failures = value.get("failures")
                if not isinstance(failures, list):
                    invalid.append(f"{rel}.{section}.failures: not list")
                elif failures:
                    invalid.append(f"{rel}.{section}.failures: nonempty")
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
                required = SUBSPACE_REQUIRED_FILES
                required_summary = SUBSPACE_REQUIRED_SUMMARY
                required_positive = ("population", "candidates_per_sec", "prompts_per_sec", "output_tokens_per_sec")
                required_finite = ("lazy_overhead_pct",)
                required_nonempty = (
                    "basis_hash",
                    "target_set_hash",
                    "basis_collection_config_hash",
                    "subspace_state_hash",
                    "candidate_scores_hash",
                    "scorer_name",
                    "scorer_version",
                    "prompt_ids_hash",
                    "sample_set_hash",
                    "prompt_scoring_config_hash",
                    "decode_config_hash",
                )
                required_path_keys = ()
                expected_summary_values = {
                    "kind": f"subspace_{spec.backend}_search",
                    "backend": spec.backend,
                    "method": "subspace",
                    "scale_mode": config.scale_mode,
                    "budget_policy": config.budget_policy,
                    "candidate_routing": "row_candidate_id",
                    "prefix_cache_policy": "disabled-for-search",
                }
                required_jsonl_nonempty = ("candidates.jsonl", "candidate_scores.jsonl")
                required_jsonl_fields = {
                    "candidates.jsonl": SUBSPACE_CANDIDATE_FIELDS,
                    "candidate_scores.jsonl": SUBSPACE_CANDIDATE_SCORE_FIELDS,
                }
                expected_jsonl_counts = {"candidates.jsonl": "population"}
                required_bool_keys = ()
                required_json_fields = {
                    "subspace_state_summary.json": ("schema_version", "basis_hash", "activation_sites", "targets"),
                    "top_k_ensemble.json": SUBSPACE_TOP_K_FIELDS,
                    "validation_report.json": SUBSPACE_VALIDATION_SECTIONS,
                    "systems_report.json": SUBSPACE_SYSTEMS_FIELDS,
                }
                required_json_nonempty_fields = {
                    "subspace_state_summary.json": ("basis_hash", "activation_sites", "targets"),
                    "top_k_ensemble.json": ("candidates", "basis_hash", "target_set_hash", "scorer_version", "prompt_ids_hash"),
                    "systems_report.json": ("gpu_model", "candidate_batch_size", "diversity_metrics"),
                }
                expected_json_values = {
                    "systems_report.json": {"prefix_cache_policy": "disabled-for-search"},
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
                expected_json_values = {}
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
                expected_json_values=expected_json_values if spec.kind == "search" else None,
            )
        )
    if config.method == "subspace":
        systems_csvs = []
        systems_files = ["report.md", "systems_report.json"]
        systems_json_fields = {"systems_report.json": SUBSPACE_SYSTEMS_FIELDS}
        systems_expected_json_values = {"systems_report.json": {"prefix_cache_policy": "disabled-for-search"}}
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
        systems_expected_json_values = {}
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
            expected_json_values=systems_expected_json_values,
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
