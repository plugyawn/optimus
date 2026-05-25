from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path

from optimus.runs.gpu_suite import GpuSuiteConfig, add_config_args, config_from_args, gpu_suite_specs, parse_int_tuple


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
    "model_id_or_path",
    "model_revision",
    "tokenizer_hash",
    "task_config_hash",
    "prompt_contract_hash",
    "screen_split_hash",
    "holdout_split_hash",
    "screen_holdout_overlap",
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
    "kernel",
    "resolved_target_scales",
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
    "selection_stage",
    "selection_rule_hash",
    "promoted_by_candidate_id",
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
    "scientific_gate_contract",
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
    "timing_evidence_paths",
)
SUBSPACE_SYSTEMS_AXIS_FIELDS = (
    "benchmark_kind",
    "population",
    "target_preset",
    "basis_rank",
    "kernel",
)
SUBSPACE_JSON_PROVENANCE_FIELDS = (
    "schema_version",
    "created_at",
    "optimus_version",
    "git_commit",
    "git_dirty",
    "command",
    "environment",
    "model_id_or_path",
    "model_revision",
    "tokenizer_hash",
    "task_config_hash",
    "prompt_contract_hash",
    "screen_split_hash",
    "holdout_split_hash",
    "decode_config_hash",
)
SUBSPACE_STATE_SUMMARY_FIELDS = (
    *SUBSPACE_JSON_PROVENANCE_FIELDS,
    "basis_hash",
    "target_preset",
    "explicit_targets",
    "layers",
    "basis_kind",
    "basis_centering",
    "basis_token_source",
    "basis_split",
    "activation_sites",
    "targets",
)
SUBSPACE_ACTIVATION_SITE_FIELDS = (
    "site_id",
    "architecture_family",
    "layer_index",
    "block_path",
    "read_tensor_path",
    "hook_point",
    "norm_position",
    "shape_convention",
    "runtime_dtype",
    "accumulation_dtype",
    "tensor_parallel_sharding_policy",
    "target_module_ids",
    "calibration_prompt_ids_hash",
    "calibration_decode_config_hash",
    "basis_control_seed",
    "transductive",
    "input_dim",
    "basis_kind",
    "requested_rank",
    "effective_rank",
    "basis_tensor_key",
    "basis_tensor_sha256",
    "singular_values",
    "captured_energy",
    "prefill_captured_energy",
    "decode_captured_energy",
    "H_s",
    "A_s",
    "orthonormality_error",
    "gram_error",
    "num_calibration_tokens",
)
SUBSPACE_TARGET_FIELDS = (
    "target_id",
    "activation_site_id",
    "output_dim",
    "base_output_power_P_t",
)
SUBSPACE_EXPECTED_SCHEMAS = {
    "summary.json": "subspace_run_summary_v1",
    "subspace_state_summary.json": "subspace_state_v1",
    "top_k_ensemble.json": "top_k_ensemble_v1",
    "validation_report.json": "validation_report_v1",
    "systems_report.json": "subspace_systems_report_v1",
}


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
SUBSPACE_SYSTEMS_NUMERIC_FIELDS = (
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
    "top_k_ensemble_cost_multiplier",
    "screen_score",
    "holdout_score",
    "screen_to_holdout_drop",
)


def _is_finite_number(value: object) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return isfinite(numeric)


def _is_json_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(float(value))


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _number_in_grid(value: object, grid: object) -> bool:
    if not _is_json_number(value) or not isinstance(grid, list):
        return False
    target = float(value)
    return any(_is_json_number(item) and abs(float(item) - target) <= 1e-12 for item in grid)


def _numeric_grids_equal(left: object, right: object) -> bool:
    if not isinstance(left, list) or not isinstance(right, list) or len(left) != len(right):
        return False
    if not all(_is_json_number(item) for item in left + right):
        return False
    return sorted(float(item) for item in left) == sorted(float(item) for item in right)


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
    if not isinstance(row.get("candidate_id"), str) or not row.get("candidate_id"):
        invalid.append(f"{prefix}.candidate_id: empty")
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
    if not isinstance(row.get("candidate_id"), str) or not row.get("candidate_id"):
        invalid.append(f"{prefix}.candidate_id: empty")
    if row.get("split") not in {"screen", "holdout", "validation", "test"}:
        invalid.append(f"{prefix}.split: invalid")
    if not isinstance(row.get("selection_stage"), str) or not row.get("selection_stage"):
        invalid.append(f"{prefix}.selection_stage: empty")
    if not isinstance(row.get("selection_rule_hash"), str) or not row.get("selection_rule_hash"):
        invalid.append(f"{prefix}.selection_rule_hash: empty")
    if row.get("split") == "holdout" and not row.get("promoted_by_candidate_id"):
        invalid.append(f"{prefix}.promoted_by_candidate_id: required for selected holdout rows")
    if not isinstance(row.get("aggregate_metrics"), dict) or not row.get("aggregate_metrics"):
        invalid.append(f"{prefix}.aggregate_metrics: empty")
    if not _is_positive_int(row.get("sample_count")):
        invalid.append(f"{prefix}.sample_count: not positive integer")
    for field in ("elapsed_s", "output_tokens"):
        if not _is_finite_number(row.get(field)):
            invalid.append(f"{prefix}.{field}: not finite")
    for field in ("scorer_name", "scorer_version", "prompt_ids_hash", "sample_set_hash", "decode_config_hash"):
        expected = summary.get(field)
        if expected is not None and row.get(field) != expected:
            invalid.append(f"{prefix}.{field}: expected summary.{field} {expected!r}, got {row.get(field)!r}")


def _evidence_path_exists(root: Path, value: str) -> bool:
    return _path_under_root(root, value) is not None


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _torch_tensor_sha256(tensor: object) -> str:
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("not a torch tensor")
    tensor = tensor.detach().cpu().contiguous()
    header = json.dumps(
        {
            "schema_version": "tensor_sha256_v2",
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(header + b"\n" + bytes(tensor.untyped_storage())).hexdigest()


def _path_under_root(root: Path, value: object) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    candidate = path if path.is_absolute() else root / path
    try:
        resolved_root = root.resolve()
        resolved_candidate = candidate.resolve()
        resolved_candidate.relative_to(resolved_root)
    except (OSError, ValueError):
        return None
    return resolved_candidate if resolved_candidate.exists() else None


def _read_hashed_json_artifact(invalid: list[str], rel: str, prefix: str, root: Path, path_value: object, hash_value: object) -> dict | None:
    if not isinstance(path_value, str) or not path_value:
        invalid.append(f"{rel}.scientific_gate_contract.{prefix}_path: empty")
        return None
    if not isinstance(hash_value, str) or not hash_value:
        invalid.append(f"{rel}.scientific_gate_contract.{prefix}_hash: empty")
        return None
    path = _path_under_root(root, path_value)
    if path is None:
        invalid.append(f"{rel}.scientific_gate_contract.{prefix}_path: missing or outside run bundle")
        return None
    observed_hash = _sha256_path(path)
    if observed_hash != hash_value:
        invalid.append(f"{rel}.scientific_gate_contract.{prefix}_hash: does not match artifact")
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        invalid.append(f"{rel}.scientific_gate_contract.{prefix}_path: artifact is not JSON")
        return None
    if not isinstance(payload, dict):
        invalid.append(f"{rel}.scientific_gate_contract.{prefix}_path: artifact is not an object")
        return None
    return payload


def _check_subspace_state_payload(invalid: list[str], root: Path, summary: dict, state_summary: dict) -> None:
    path = root / "subspace_state.pt"
    if not path.exists():
        return
    try:
        import torch

        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
    except Exception as exc:  # pragma: no cover - exact torch exception varies.
        invalid.append(f"subspace_state.pt: cannot load payload: {exc}")
        return
    if not isinstance(payload, dict):
        invalid.append("subspace_state.pt: payload is not a dict")
        return
    if payload.get("schema_version") != "subspace_state_payload_v1":
        invalid.append(f"subspace_state.pt.schema_version: expected 'subspace_state_payload_v1', got {payload.get('schema_version')!r}")
    tensors = payload.get("basis_tensors")
    if not isinstance(tensors, dict) or not tensors:
        invalid.append("subspace_state.pt.basis_tensors: empty")
        return
    for index, site in enumerate(state_summary.get("activation_sites") or [], start=1):
        if not isinstance(site, dict):
            continue
        key = site.get("basis_tensor_key")
        expected_tensor_hash = site.get("basis_tensor_sha256")
        if not isinstance(key, str) or not key:
            continue
        if key not in tensors:
            invalid.append(f"subspace_state.pt.basis_tensors: missing {key!r}")
            continue
        if not isinstance(expected_tensor_hash, str) or not expected_tensor_hash:
            invalid.append(f"subspace_state_summary.json.activation_sites[{index}].basis_tensor_sha256: empty")
            continue
        try:
            observed_tensor_hash = _torch_tensor_sha256(tensors[key])
        except Exception as exc:  # pragma: no cover - exact tensor exception varies.
            invalid.append(f"subspace_state.pt.basis_tensors[{key!r}]: cannot hash tensor: {exc}")
            continue
        if observed_tensor_hash != expected_tensor_hash:
            invalid.append(f"subspace_state.pt.basis_tensors[{key!r}]: sha256 mismatch")


def _timing_evidence_has_sync_marker(path: Path) -> bool:
    try:
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if isinstance(row, dict) and row.get("cuda_synchronized") is True:
                    return True
    except (OSError, json.JSONDecodeError):
        return False
    return False


def _evidence_has_substantive_payload(payload: dict) -> bool:
    for key in ("checks", "metrics", "artifacts"):
        value = payload.get(key)
        if isinstance(value, dict) and value:
            return True
        if isinstance(value, list) and value:
            return True
    return False


def _check_validation_evidence_payload(invalid: list[str], rel: str, section: str, evidence: str, payload: object) -> None:
    if not isinstance(payload, dict):
        invalid.append(f"{rel}.{section}.evidence_paths: evidence is not an object {evidence!r}")
        return
    if payload.get("evidence_schema_version") != "validation_evidence_v1":
        invalid.append(f"{rel}.{section}.evidence_paths: evidence_schema_version missing or invalid {evidence!r}")
    if payload.get("section") != section:
        invalid.append(f"{rel}.{section}.evidence_paths: evidence section mismatch {evidence!r}")
    if payload.get("status") != "pass":
        invalid.append(f"{rel}.{section}.evidence_paths: evidence status is not pass {evidence!r}")
    if not isinstance(payload.get("generated_at"), str) or not payload.get("generated_at"):
        invalid.append(f"{rel}.{section}.evidence_paths: generated_at missing {evidence!r}")
    if not isinstance(payload.get("command"), list) or not payload.get("command"):
        invalid.append(f"{rel}.{section}.evidence_paths: command missing {evidence!r}")
    if not _evidence_has_substantive_payload(payload):
        invalid.append(f"{rel}.{section}.evidence_paths: no substantive evidence payload {evidence!r}")
    if section == "drift_diagnostics":
        _check_drift_evidence_payload(invalid, rel, section, evidence, payload)


def _check_drift_evidence_payload(invalid: list[str], rel: str, section: str, evidence: str, payload: dict) -> None:
    for field in ("probe_split_hash", "reference_artifact_hash", "candidate_artifact_hash", "aggregation"):
        if not isinstance(payload.get(field), str) or not payload.get(field):
            invalid.append(f"{rel}.{section}.evidence_paths: {field} missing {evidence!r}")
    if not _is_positive_int(payload.get("sample_count")):
        invalid.append(f"{rel}.{section}.evidence_paths: sample_count not positive integer {evidence!r}")
    for field in ("temperature", "epsilon"):
        if not _is_json_number(payload.get(field)):
            invalid.append(f"{rel}.{section}.evidence_paths: {field} not JSON number {evidence!r}")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        invalid.append(f"{rel}.{section}.evidence_paths: metrics missing {evidence!r}")
        return
    for field in ("logit_kl_mean", "hidden_state_rms_drift"):
        if not _is_json_number(metrics.get(field)):
            invalid.append(f"{rel}.{section}.evidence_paths: metrics.{field} not JSON number {evidence!r}")


ENGINEERING_ADVANTAGE_THRESHOLDS = {
    "logit_kl_mean_reduction_pct": 25.0,
    "hidden_state_rms_drift_reduction_pct": 25.0,
    "lazy_overhead_reduction_pct": 20.0,
    "captured_energy_gain_pct_points": 10.0,
}


def _check_subspace_state_summary(invalid: list[str], rel: str, payload: dict) -> None:
    if payload.get("basis_kind") not in {"activation-svd", "random-orthonormal", "shuffled-activation-svd"}:
        invalid.append(f"{rel}.basis_kind: invalid")
    if payload.get("basis_centering") not in {"none", "mean"}:
        invalid.append(f"{rel}.basis_centering: invalid")
    if payload.get("basis_token_source") not in {"prefill", "decode", "prefill+decode"}:
        invalid.append(f"{rel}.basis_token_source: invalid")
    if payload.get("basis_split") not in {"train", "screen_unlabeled", "public_unlabeled", "holdout_forbidden"}:
        invalid.append(f"{rel}.basis_split: invalid")
    sites = payload.get("activation_sites")
    if isinstance(sites, list) and sites:
        for index, site in enumerate(sites, start=1):
            if not isinstance(site, dict):
                invalid.append(f"{rel}.activation_sites[{index}]: not object")
                continue
            for field in SUBSPACE_ACTIVATION_SITE_FIELDS:
                if field not in site:
                    invalid.append(f"{rel}.activation_sites[{index}].{field}: missing")
            for field in ("input_dim", "requested_rank", "effective_rank", "num_calibration_tokens"):
                if field in site and not _is_positive_int(site.get(field)):
                    invalid.append(f"{rel}.activation_sites[{index}].{field}: not positive integer")
            if "layer_index" in site and not _is_nonnegative_int(site.get("layer_index")):
                invalid.append(f"{rel}.activation_sites[{index}].layer_index: not nonnegative integer")
            for field in ("captured_energy", "H_s", "A_s", "orthonormality_error", "gram_error"):
                if field in site and not _is_finite_number(site.get(field)):
                    invalid.append(f"{rel}.activation_sites[{index}].{field}: not finite")
            for field in ("target_module_ids",):
                if field in site and (not isinstance(site.get(field), list) or not site.get(field)):
                    invalid.append(f"{rel}.activation_sites[{index}].{field}: empty")
            if "transductive" in site and not isinstance(site.get("transductive"), bool):
                invalid.append(f"{rel}.activation_sites[{index}].transductive: not boolean")
    targets = payload.get("targets")
    if isinstance(targets, list) and targets:
        for index, target in enumerate(targets, start=1):
            if not isinstance(target, dict):
                invalid.append(f"{rel}.targets[{index}]: not object")
                continue
            for field in SUBSPACE_TARGET_FIELDS:
                if field not in target:
                    invalid.append(f"{rel}.targets[{index}].{field}: missing")
            if "output_dim" in target and not _is_positive_int(target.get("output_dim")):
                invalid.append(f"{rel}.targets[{index}].output_dim: not positive integer")
            if "base_output_power_P_t" in target and not _is_finite_number(target.get("base_output_power_P_t")):
                invalid.append(f"{rel}.targets[{index}].base_output_power_P_t: not finite")


def _check_subspace_systems(invalid: list[str], rel: str, payload: dict, root: Path, *, suite_report: bool = False) -> None:
    for field in SUBSPACE_SYSTEMS_NUMERIC_FIELDS:
        if field in payload and not _is_json_number(payload.get(field)):
            invalid.append(f"{rel}.{field}: not JSON number")
    for field in SUBSPACE_SYSTEMS_AXIS_FIELDS:
        if field not in payload or payload.get(field) in {None, ""}:
            invalid.append(f"{rel}.{field}: empty")
    source_base = root
    if suite_report:
        for field in ("source_report", "source_run_dir"):
            if not isinstance(payload.get(field), str) or not payload.get(field):
                invalid.append(f"{rel}.{field}: empty")
        source_report = payload.get("source_report")
        if isinstance(source_report, str) and source_report:
            source_report_path = Path(source_report)
            if not source_report_path.is_absolute():
                source_report_path = root / source_report_path
            if not source_report_path.exists():
                invalid.append(f"{rel}.source_report: missing {source_report!r}")
        source_run_dir = payload.get("source_run_dir")
        if isinstance(source_run_dir, str) and source_run_dir:
            source_base = Path(source_run_dir)
            if not source_base.is_absolute():
                source_base = root / source_base
            if not source_base.exists():
                invalid.append(f"{rel}.source_run_dir: missing {source_run_dir!r}")
    for field in ("diversity_metrics", "random_q_control", "shuffled_q_control", "antithetic_odd_even"):
        if field in payload and not isinstance(payload.get(field), dict):
            invalid.append(f"{rel}.{field}: not object")
    if payload.get("prefix_cache_policy") != "disabled-for-search":
        invalid.append(f"{rel}.prefix_cache_policy: expected 'disabled-for-search', got {payload.get('prefix_cache_policy')!r}")
    timing_evidence_paths = payload.get("timing_evidence_paths")
    if not isinstance(timing_evidence_paths, list) or not timing_evidence_paths:
        invalid.append(f"{rel}.timing_evidence_paths: empty")
    else:
        for evidence in timing_evidence_paths:
            if not isinstance(evidence, str) or not evidence:
                invalid.append(f"{rel}.timing_evidence_paths: missing {evidence!r}")
                continue
            evidence_path = Path(evidence)
            if not evidence_path.is_absolute():
                evidence_path = source_base / evidence_path
            if not evidence_path.exists():
                invalid.append(f"{rel}.timing_evidence_paths: missing {evidence!r}")
            elif not _timing_evidence_has_sync_marker(evidence_path):
                invalid.append(f"{rel}.timing_evidence_paths: no cuda_synchronized marker in {evidence!r}")
    if payload.get("population") == 128:
        base_time = payload.get("base_model_time_s")
        qx_time = payload.get("qx_time_s")
        delta_time = payload.get("lazy_delta_time_s")
        if all(_is_json_number(value) for value in (base_time, qx_time, delta_time)) and float(base_time) > 0:
            overhead = (float(qx_time) + float(delta_time)) / float(base_time)
            if overhead > 0.25:
                invalid.append(f"{rel}.p128_speed_gate: qx_plus_lazy_delta_overhead {overhead:.3f} > 0.25")
        else:
            invalid.append(f"{rel}.p128_speed_gate: missing timed base/qx/lazy fields")


def _check_scientific_gate_contract(
    invalid: list[str],
    rel: str,
    section: dict,
    *,
    root: Path,
    summary: dict,
    top_k: dict | None,
    selection_rule_hashes: set[str],
) -> None:
    allowed_corrections = {
        "none_predeclared_single_config",
        "holm_bonferroni",
        "bonferroni",
        "benjamini_hochberg",
        "separate_validation_split",
    }
    for field in (
        "gate_stage",
        "locked_config_hash",
        "selection_rule_hash",
        "primary_metric",
        "multiple_comparison_correction",
        "basis_kind",
        "comparison",
        "gate_type",
        "locked_target_preset",
        "locked_scale_mode",
        "locked_aggregation",
        "selection_split",
    ):
        if not isinstance(section.get(field), str) or not section.get(field):
            invalid.append(f"{rel}.scientific_gate_contract.{field}: empty")
    for field in ("locked_K", "locked_basis_rank"):
        if not _is_positive_int(section.get(field)):
            invalid.append(f"{rel}.scientific_gate_contract.{field}: not positive integer")
    for field in ("K_grid", "basis_rank_grid"):
        values = section.get(field)
        if not isinstance(values, list) or not values or not all(_is_positive_int(item) for item in values):
            invalid.append(f"{rel}.scientific_gate_contract.{field}: invalid")
    radius_grid = section.get("radius_grid")
    if not isinstance(radius_grid, list) or not radius_grid or not all(_is_json_number(item) for item in radius_grid):
        invalid.append(f"{rel}.scientific_gate_contract.radius_grid: invalid")
    if not _is_finite_number(section.get("locked_radius")):
        invalid.append(f"{rel}.scientific_gate_contract.locked_radius: not finite")
    if isinstance(section.get("K_grid"), list) and _is_positive_int(section.get("locked_K")) and section["locked_K"] not in section["K_grid"]:
        invalid.append(f"{rel}.scientific_gate_contract.locked_K: not present in K_grid")
    if (
        isinstance(section.get("basis_rank_grid"), list)
        and _is_positive_int(section.get("locked_basis_rank"))
        and section["locked_basis_rank"] not in section["basis_rank_grid"]
    ):
        invalid.append(f"{rel}.scientific_gate_contract.locked_basis_rank: not present in basis_rank_grid")
    if isinstance(radius_grid, list) and _is_finite_number(section.get("locked_radius")) and not _number_in_grid(section["locked_radius"], radius_grid):
        invalid.append(f"{rel}.scientific_gate_contract.locked_radius: not present in radius_grid")
    correction = section.get("multiple_comparison_correction")
    if correction not in allowed_corrections:
        invalid.append(f"{rel}.scientific_gate_contract.multiple_comparison_correction: invalid {correction!r}")
    grid_lengths = [
        len(values)
        for values in (section.get("K_grid"), section.get("basis_rank_grid"), radius_grid)
        if isinstance(values, list) and values
    ]
    multi_config_grid = any(length > 1 for length in grid_lengths)
    if multi_config_grid and correction == "none_predeclared_single_config":
        invalid.append(f"{rel}.scientific_gate_contract.multiple_comparison_correction: none_predeclared_single_config requires singleton grids")
    if correction == "separate_validation_split":
        for field in ("validation_selection_split_hash", "validation_selection_artifact_hash"):
            if not isinstance(section.get(field), str) or not section.get(field):
                invalid.append(f"{rel}.scientific_gate_contract.{field}: required for separate_validation_split")
        if section.get("validation_selection_split_hash") in {summary.get("screen_split_hash"), summary.get("holdout_split_hash")}:
            invalid.append(f"{rel}.scientific_gate_contract.validation_selection_split_hash: must differ from screen and holdout split hashes")
        validation_artifact = _read_hashed_json_artifact(
            invalid,
            rel,
            "validation_selection_artifact",
            root,
            section.get("validation_selection_artifact_path"),
            section.get("validation_selection_artifact_hash"),
        )
        if validation_artifact is not None and validation_artifact.get("schema_version") != "validation_selection_artifact_v1":
            invalid.append(f"{rel}.scientific_gate_contract.validation_selection_artifact_path: invalid schema_version")
        if validation_artifact is not None:
            if validation_artifact.get("selection_split_hash") != section.get("validation_selection_split_hash"):
                invalid.append(f"{rel}.scientific_gate_contract.validation_selection_artifact.selection_split_hash: does not match contract")
            if validation_artifact.get("selection_rule_hash") != section.get("selection_rule_hash"):
                invalid.append(f"{rel}.scientific_gate_contract.validation_selection_artifact.selection_rule_hash: does not match contract")
    observed_corrected_contrast_keys: set[tuple[object, ...]] = set()
    gate_family_artifact = _read_hashed_json_artifact(
        invalid,
        rel,
        "gate_family_artifact",
        root,
        section.get("gate_family_artifact_path"),
        section.get("gate_family_artifact_hash"),
    )
    if gate_family_artifact is not None:
        if gate_family_artifact.get("schema_version") != "scientific_gate_family_v1":
            invalid.append(f"{rel}.scientific_gate_contract.gate_family_artifact_path: invalid schema_version")
        for field in ("primary_metric", "multiple_comparison_correction", "selection_rule_hash", "holdout_tuned"):
            if gate_family_artifact.get(field) != section.get(field):
                invalid.append(f"{rel}.scientific_gate_contract.gate_family_artifact.{field}: does not match contract")
        for field in ("K_grid", "basis_rank_grid", "radius_grid"):
            if not _numeric_grids_equal(section.get(field), gate_family_artifact.get(field)):
                invalid.append(f"{rel}.scientific_gate_contract.gate_family_artifact.{field}: does not match contract")
        observed_configs = gate_family_artifact.get("observed_configs")
        if not isinstance(observed_configs, list) or not observed_configs:
            invalid.append(f"{rel}.scientific_gate_contract.gate_family_artifact.observed_configs: empty")
        else:
            expected_k_grid = section.get("K_grid") if isinstance(section.get("K_grid"), list) else []
            expected_rank_grid = section.get("basis_rank_grid") if isinstance(section.get("basis_rank_grid"), list) else []
            expected_radius_grid = section.get("radius_grid") if isinstance(section.get("radius_grid"), list) else []
            observed_k = sorted({config.get("K") for config in observed_configs if isinstance(config, dict) and _is_positive_int(config.get("K"))})
            observed_rank = sorted({config.get("basis_rank") for config in observed_configs if isinstance(config, dict) and _is_positive_int(config.get("basis_rank"))})
            observed_radius = sorted({float(config.get("radius")) for config in observed_configs if isinstance(config, dict) and _is_json_number(config.get("radius"))})
            if observed_k != sorted(expected_k_grid):
                invalid.append(f"{rel}.scientific_gate_contract.gate_family_artifact.observed_configs: K_grid mismatch")
            if observed_rank != sorted(expected_rank_grid):
                invalid.append(f"{rel}.scientific_gate_contract.gate_family_artifact.observed_configs: basis_rank_grid mismatch")
            if observed_radius != sorted(float(item) for item in expected_radius_grid if _is_json_number(item)):
                invalid.append(f"{rel}.scientific_gate_contract.gate_family_artifact.observed_configs: radius_grid mismatch")
            observed_basis = {config.get("basis_kind") for config in observed_configs if isinstance(config, dict)}
            required_basis = {"activation-svd", "random-orthonormal", "shuffled-activation-svd"}
            if required_basis - observed_basis:
                invalid.append(f"{rel}.scientific_gate_contract.gate_family_artifact.observed_configs: missing basis families {sorted(required_basis - observed_basis)!r}")
            observed_basis_by_config: dict[tuple[object, ...], set[str]] = {}
            for config in observed_configs:
                if not isinstance(config, dict) or not _is_positive_int(config.get("K")) or not _is_positive_int(config.get("basis_rank")):
                    continue
                if not _is_json_number(config.get("radius")):
                    continue
                key = (
                    config.get("K"),
                    config.get("basis_rank"),
                    float(config.get("radius")),
                    config.get("target_preset"),
                    config.get("scale_mode"),
                    config.get("aggregation"),
                )
                observed_basis_by_config.setdefault(key, set()).add(str(config.get("basis_kind")))
            for key, basis_kinds in observed_basis_by_config.items():
                if "activation-svd" not in basis_kinds:
                    continue
                for control_basis_kind in ("random-orthonormal", "shuffled-activation-svd"):
                    if control_basis_kind in basis_kinds:
                        observed_corrected_contrast_keys.add((control_basis_kind, *key))
            locked_seen = any(
                isinstance(config, dict)
                and config.get("basis_kind") == section.get("basis_kind")
                and config.get("K") == section.get("locked_K")
                and config.get("basis_rank") == section.get("locked_basis_rank")
                and _is_json_number(config.get("radius"))
                and _is_json_number(section.get("locked_radius"))
                and abs(float(config.get("radius")) - float(section.get("locked_radius"))) <= 1e-12
                and config.get("target_preset") == section.get("locked_target_preset")
                and config.get("scale_mode") == section.get("locked_scale_mode")
                and config.get("aggregation") == section.get("locked_aggregation")
                for config in observed_configs
            )
            if not locked_seen:
                invalid.append(f"{rel}.scientific_gate_contract.gate_family_artifact.observed_configs: locked config missing")
            for index, config in enumerate(observed_configs, start=1):
                if not isinstance(config, dict):
                    invalid.append(f"{rel}.scientific_gate_contract.gate_family_artifact.observed_configs[{index}]: not object")
                    continue
                config_artifact = _read_hashed_json_artifact(
                    invalid,
                    rel,
                    f"gate_family_artifact.observed_configs[{index}].artifact",
                    root,
                    config.get("artifact_path"),
                    config.get("artifact_hash"),
                )
                if config_artifact is None:
                    continue
                if config_artifact.get("schema_version") != "scientific_gate_config_v1":
                    invalid.append(f"{rel}.scientific_gate_contract.gate_family_artifact.observed_configs[{index}].artifact_path: invalid schema_version")
                for field in ("basis_kind", "K", "basis_rank", "target_preset", "scale_mode", "aggregation"):
                    if config_artifact.get(field) != config.get(field):
                        invalid.append(f"{rel}.scientific_gate_contract.gate_family_artifact.observed_configs[{index}].artifact.{field}: does not match observed config")
                if _is_json_number(config_artifact.get("radius")) and _is_json_number(config.get("radius")):
                    if abs(float(config_artifact["radius"]) - float(config["radius"])) > 1e-12:
                        invalid.append(f"{rel}.scientific_gate_contract.gate_family_artifact.observed_configs[{index}].artifact.radius: does not match observed config")
                else:
                    invalid.append(f"{rel}.scientific_gate_contract.gate_family_artifact.observed_configs[{index}].artifact.radius: does not match observed config")
                if config_artifact.get("primary_metric") != section.get("primary_metric"):
                    invalid.append(f"{rel}.scientific_gate_contract.gate_family_artifact.observed_configs[{index}].artifact.primary_metric: does not match contract")
                if config_artifact.get("selection_rule_hash") != section.get("selection_rule_hash"):
                    invalid.append(f"{rel}.scientific_gate_contract.gate_family_artifact.observed_configs[{index}].artifact.selection_rule_hash: does not match contract")
    if section.get("selection_split") != "screen":
        invalid.append(f"{rel}.scientific_gate_contract.selection_split: expected 'screen'")
    if section.get("holdout_tuned") is not False:
        invalid.append(f"{rel}.scientific_gate_contract.holdout_tuned: expected false")
    if not (isinstance(section.get("screen_holdout_overlap"), int) and not isinstance(section.get("screen_holdout_overlap"), bool) and section.get("screen_holdout_overlap") == 0):
        invalid.append(f"{rel}.scientific_gate_contract.screen_holdout_overlap: expected 0")
    if section.get("selection_rule_hash") not in selection_rule_hashes:
        invalid.append(f"{rel}.scientific_gate_contract.selection_rule_hash: not present in candidate_scores")
    if top_k is not None:
        if section.get("locked_config_hash") != top_k.get("runtime_config_hash"):
            invalid.append(f"{rel}.scientific_gate_contract.locked_config_hash: does not match top_k runtime_config_hash")
        if section.get("locked_K") != top_k.get("K"):
            invalid.append(f"{rel}.scientific_gate_contract.locked_K: does not match top_k K")
        if section.get("locked_radius") != top_k.get("rho_or_sigma_w"):
            invalid.append(f"{rel}.scientific_gate_contract.locked_radius: does not match top_k rho_or_sigma_w")
        if section.get("locked_scale_mode") != top_k.get("scale_mode"):
            invalid.append(f"{rel}.scientific_gate_contract.locked_scale_mode: does not match top_k scale_mode")
        if section.get("locked_aggregation") != top_k.get("aggregation"):
            invalid.append(f"{rel}.scientific_gate_contract.locked_aggregation: does not match top_k aggregation")
        candidates = top_k.get("candidates")
        if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict):
            for index, candidate in enumerate(candidates, start=1):
                if not isinstance(candidate, dict):
                    continue
                if section.get("locked_basis_rank") != candidate.get("basis_rank"):
                    invalid.append(f"{rel}.scientific_gate_contract.locked_basis_rank: does not match top_k candidate[{index}] basis_rank")
                if section.get("locked_target_preset") != candidate.get("target_preset"):
                    invalid.append(f"{rel}.scientific_gate_contract.locked_target_preset: does not match top_k candidate[{index}] target_preset")
                if section.get("locked_scale_mode") != candidate.get("scale_mode"):
                    invalid.append(f"{rel}.scientific_gate_contract.locked_scale_mode: does not match top_k candidate[{index}] scale_mode")
                if _is_json_number(section.get("locked_radius")) and _is_json_number(candidate.get("rho_or_sigma_w")):
                    if abs(float(section["locked_radius"]) - float(candidate["rho_or_sigma_w"])) > 1e-12:
                        invalid.append(f"{rel}.scientific_gate_contract.locked_radius: does not match top_k candidate[{index}] rho_or_sigma_w")
                else:
                    invalid.append(f"{rel}.scientific_gate_contract.locked_radius: does not match top_k candidate[{index}] rho_or_sigma_w")
    if section.get("screen_holdout_overlap") != summary.get("screen_holdout_overlap"):
        invalid.append(f"{rel}.scientific_gate_contract.screen_holdout_overlap: does not match summary")
    if section.get("gate_stage") not in {"reference_smoke", "production"}:
        invalid.append(f"{rel}.scientific_gate_contract.gate_stage: invalid {section.get('gate_stage')!r}")
    if section.get("basis_kind") != "activation-svd":
        invalid.append(f"{rel}.scientific_gate_contract.basis_kind: expected 'activation-svd', got {section.get('basis_kind')!r}")
    controls = section.get("control_basis_kinds")
    if not isinstance(controls, list) or not controls or not all(isinstance(item, str) and item for item in controls):
        invalid.append(f"{rel}.scientific_gate_contract.control_basis_kinds: empty")
    if set(controls or []) != {"random-orthonormal", "shuffled-activation-svd"}:
        invalid.append(f"{rel}.scientific_gate_contract.control_basis_kinds: must include exactly random-orthonormal and shuffled-activation-svd")
    control_artifacts = section.get("compared_control_artifact_hashes")
    if not isinstance(control_artifacts, dict):
        invalid.append(f"{rel}.scientific_gate_contract.compared_control_artifact_hashes: not object")
    else:
        for basis_kind in ("random-orthonormal", "shuffled-activation-svd"):
            if not isinstance(control_artifacts.get(basis_kind), str) or not control_artifacts.get(basis_kind):
                invalid.append(f"{rel}.scientific_gate_contract.compared_control_artifact_hashes.{basis_kind}: empty")
    control_paths = section.get("compared_control_artifact_paths")
    if not isinstance(control_paths, dict):
        invalid.append(f"{rel}.scientific_gate_contract.compared_control_artifact_paths: not object")
    else:
        for basis_kind in ("random-orthonormal", "shuffled-activation-svd"):
            expected_hash = control_artifacts.get(basis_kind) if isinstance(control_artifacts, dict) else None
            control_payload = _read_hashed_json_artifact(
                invalid,
                rel,
                f"compared_control_artifact_paths.{basis_kind}",
                root,
                control_paths.get(basis_kind),
                expected_hash,
            )
            if control_payload is None:
                continue
            if control_payload.get("schema_version") != "scientific_gate_control_v1":
                invalid.append(f"{rel}.scientific_gate_contract.compared_control_artifact_paths.{basis_kind}: invalid schema_version")
            if control_payload.get("basis_kind") != basis_kind:
                invalid.append(f"{rel}.scientific_gate_contract.compared_control_artifact_paths.{basis_kind}: basis_kind mismatch")
            if control_payload.get("metric") != section.get("primary_metric"):
                invalid.append(f"{rel}.scientific_gate_contract.compared_control_artifact_paths.{basis_kind}: metric mismatch")
    contrasts = section.get("tested_contrasts")
    if not isinstance(contrasts, list) or not contrasts:
        invalid.append(f"{rel}.scientific_gate_contract.tested_contrasts: empty")
    else:
        covered_controls: set[str] = set()
        covered_corrected_contrast_keys: set[tuple[object, ...]] = set()
        primary_metric = section.get("primary_metric")
        for index, contrast in enumerate(contrasts, start=1):
            if not isinstance(contrast, dict):
                invalid.append(f"{rel}.scientific_gate_contract.tested_contrasts[{index}]: not object")
                continue
            for field in (
                "basis_kind",
                "control_basis_kind",
                "metric",
                "artifact_path",
                "artifact_hash",
                "control_artifact_path",
                "control_artifact_hash",
                "target_preset",
                "scale_mode",
                "aggregation",
            ):
                if not isinstance(contrast.get(field), str) or not contrast.get(field):
                    invalid.append(f"{rel}.scientific_gate_contract.tested_contrasts[{index}].{field}: empty")
            for field in ("K", "basis_rank"):
                if not _is_positive_int(contrast.get(field)):
                    invalid.append(f"{rel}.scientific_gate_contract.tested_contrasts[{index}].{field}: not positive integer")
            if not _is_json_number(contrast.get("radius")):
                invalid.append(f"{rel}.scientific_gate_contract.tested_contrasts[{index}].radius: not JSON number")
            if contrast.get("basis_kind") != "activation-svd":
                invalid.append(f"{rel}.scientific_gate_contract.tested_contrasts[{index}].basis_kind: expected activation-svd")
            control_basis_kind = contrast.get("control_basis_kind")
            if control_basis_kind not in {"random-orthonormal", "shuffled-activation-svd"}:
                invalid.append(f"{rel}.scientific_gate_contract.tested_contrasts[{index}].control_basis_kind: invalid")
            if contrast.get("metric") != primary_metric:
                invalid.append(f"{rel}.scientific_gate_contract.tested_contrasts[{index}].metric: does not match primary_metric")
            if isinstance(control_artifacts, dict) and isinstance(control_basis_kind, str) and control_basis_kind in control_artifacts:
                expected_control_hash = control_artifacts.get(control_basis_kind)
                if contrast.get("control_artifact_hash") != expected_control_hash:
                    invalid.append(
                        f"{rel}.scientific_gate_contract.tested_contrasts[{index}].control_artifact_hash: does not match compared_control_artifact_hashes"
                    )
                if isinstance(control_paths, dict) and contrast.get("control_artifact_path") != control_paths.get(control_basis_kind):
                    invalid.append(
                        f"{rel}.scientific_gate_contract.tested_contrasts[{index}].control_artifact_path: does not match compared_control_artifact_paths"
                    )
            contrast_payload = _read_hashed_json_artifact(
                invalid,
                rel,
                f"tested_contrasts[{index}].artifact",
                root,
                contrast.get("artifact_path"),
                contrast.get("artifact_hash"),
            )
            if contrast_payload is not None:
                if contrast_payload.get("schema_version") != "scientific_gate_contrast_v1":
                    invalid.append(f"{rel}.scientific_gate_contract.tested_contrasts[{index}].artifact_path: invalid schema_version")
                for field in (
                    "basis_kind",
                    "control_basis_kind",
                    "metric",
                    "control_artifact_hash",
                    "K",
                    "basis_rank",
                    "target_preset",
                    "scale_mode",
                    "aggregation",
                ):
                    if contrast_payload.get(field) != contrast.get(field):
                        invalid.append(f"{rel}.scientific_gate_contract.tested_contrasts[{index}].artifact.{field}: does not match contrast")
                if _is_json_number(contrast_payload.get("radius")) and _is_json_number(contrast.get("radius")):
                    if abs(float(contrast_payload["radius"]) - float(contrast["radius"])) > 1e-12:
                        invalid.append(f"{rel}.scientific_gate_contract.tested_contrasts[{index}].artifact.radius: does not match contrast")
                else:
                    invalid.append(f"{rel}.scientific_gate_contract.tested_contrasts[{index}].artifact.radius: does not match contrast")
            _read_hashed_json_artifact(
                invalid,
                rel,
                f"tested_contrasts[{index}].control_artifact",
                root,
                contrast.get("control_artifact_path"),
                contrast.get("control_artifact_hash"),
            )
            if contrast.get("basis_kind") == "activation-svd" and contrast.get("metric") == primary_metric and control_basis_kind in {
                "random-orthonormal",
                "shuffled-activation-svd",
            }:
                covered_controls.add(str(control_basis_kind))
                if _is_positive_int(contrast.get("K")) and _is_positive_int(contrast.get("basis_rank")) and _is_json_number(contrast.get("radius")):
                    covered_corrected_contrast_keys.add(
                        (
                            str(control_basis_kind),
                            contrast.get("K"),
                            contrast.get("basis_rank"),
                            float(contrast.get("radius")),
                            contrast.get("target_preset"),
                            contrast.get("scale_mode"),
                            contrast.get("aggregation"),
                        )
                    )
        missing_contrasts = sorted({"random-orthonormal", "shuffled-activation-svd"} - covered_controls)
        if missing_contrasts:
            invalid.append(f"{rel}.scientific_gate_contract.tested_contrasts: missing primary_metric controls {missing_contrasts!r}")
        if multi_config_grid and correction in {"holm_bonferroni", "bonferroni", "benjamini_hochberg"}:
            missing_corrected = sorted(observed_corrected_contrast_keys - covered_corrected_contrast_keys, key=str)
            if missing_corrected:
                invalid.append(f"{rel}.scientific_gate_contract.tested_contrasts: missing corrected-family contrasts {missing_corrected!r}")
    if section.get("gate_type") not in {"non-inferiority", "paired-bootstrap-positive", "engineering-proceed-no-scientific-win"}:
        invalid.append(f"{rel}.scientific_gate_contract.gate_type: invalid {section.get('gate_type')!r}")
    if section.get("comparison") != "activation_svd_minus_best_control":
        invalid.append(f"{rel}.scientific_gate_contract.comparison: expected activation_svd_minus_best_control")
    if not _is_json_number(section.get("epsilon")) or float(section.get("epsilon", -1.0)) < 0:
        invalid.append(f"{rel}.scientific_gate_contract.epsilon: not nonnegative JSON number")
    ci = section.get("confidence_interval")
    if not isinstance(ci, dict):
        invalid.append(f"{rel}.scientific_gate_contract.confidence_interval: not object")
        return
    for field in ("lower", "upper"):
        if not _is_json_number(ci.get(field)):
            invalid.append(f"{rel}.scientific_gate_contract.confidence_interval.{field}: not JSON number")
    lower = ci.get("lower")
    upper = ci.get("upper")
    if _is_json_number(lower):
        gate_stage = section.get("gate_stage")
        gate_type = section.get("gate_type")
        epsilon = float(section.get("epsilon", 0.0)) if _is_json_number(section.get("epsilon")) else 0.0
        if gate_stage == "reference_smoke" and gate_type == "non-inferiority" and float(lower) < -epsilon:
            invalid.append(f"{rel}.scientific_gate_contract.confidence_interval.lower: below non-inferiority bound")
        if gate_stage == "production" and gate_type == "paired-bootstrap-positive" and float(lower) <= 0.0:
            invalid.append(f"{rel}.scientific_gate_contract.confidence_interval.lower: production gate requires lower > 0")
        if gate_stage == "production" and gate_type == "engineering-proceed-no-scientific-win":
            if float(lower) < -epsilon:
                invalid.append(f"{rel}.scientific_gate_contract.confidence_interval.lower: engineering proceed requires lower >= -epsilon")
            if float(lower) > 0.0:
                invalid.append(f"{rel}.scientific_gate_contract.confidence_interval.lower: scientific win must use paired-bootstrap-positive")
            if _is_json_number(upper) and float(upper) < 0.0:
                invalid.append(f"{rel}.scientific_gate_contract.confidence_interval.upper: engineering proceed tie interval must include zero")
            exception = section.get("engineering_exception")
            if not isinstance(exception, dict) or not exception.get("accepted_label") == "engineering_proceed_no_scientific_win":
                invalid.append(f"{rel}.scientific_gate_contract.engineering_exception: missing accepted label")
            else:
                advantage = exception.get("operational_advantage")
                if not isinstance(advantage, dict):
                    invalid.append(f"{rel}.scientific_gate_contract.engineering_exception.operational_advantage: missing")
                else:
                    metric = advantage.get("metric")
                    delta = advantage.get("delta")
                    threshold = ENGINEERING_ADVANTAGE_THRESHOLDS.get(metric)
                    if threshold is None:
                        invalid.append(f"{rel}.scientific_gate_contract.engineering_exception.operational_advantage.metric: invalid {metric!r}")
                    if not _is_json_number(delta):
                        invalid.append(f"{rel}.scientific_gate_contract.engineering_exception.operational_advantage.delta: not JSON number")
                    elif threshold is not None and float(delta) < threshold:
                        invalid.append(
                            f"{rel}.scientific_gate_contract.engineering_exception.operational_advantage.delta: {float(delta):.3f} below {threshold:.3f}"
                        )
                    for field in ("probe_split_hash", "reference_artifact_hash", "aggregation", "direction"):
                        if not isinstance(advantage.get(field), str) or not advantage.get(field):
                            invalid.append(f"{rel}.scientific_gate_contract.engineering_exception.operational_advantage.{field}: empty")


def _candidate_identity_projection(row: dict) -> dict:
    return {field: row.get(field) for field in SUBSPACE_CANDIDATE_FIELDS}


def check_run(contract: RunContract) -> RunCheck:
    missing = tuple(rel for rel in contract.required_files if not (contract.root / rel).exists())
    invalid: list[str] = []
    is_subspace_contract = "subspace_state.pt" in contract.required_files
    is_subspace_systems_contract = "subspace_systems.csv" in contract.required_files
    is_subspace_artifact_contract = is_subspace_contract or is_subspace_systems_contract
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
        expected_schema = SUBSPACE_EXPECTED_SCHEMAS["summary.json"]
        if summary.get("schema_version") != expected_schema:
            invalid.append(f"summary.schema_version: expected {expected_schema!r}, got {summary.get('schema_version')!r}")
        if not isinstance(summary.get("git_dirty"), bool):
            invalid.append("summary.git_dirty: not boolean")
        if not summary.get("command"):
            invalid.append("summary.command: empty")
        if not isinstance(summary.get("environment"), dict) or not summary.get("environment"):
            invalid.append("summary.environment: empty")
        if not (isinstance(summary.get("screen_holdout_overlap"), int) and not isinstance(summary.get("screen_holdout_overlap"), bool) and summary.get("screen_holdout_overlap") == 0):
            invalid.append("summary.screen_holdout_overlap: expected 0")
        for key in ("candidates_per_sec", "prompts_per_sec", "output_tokens_per_sec", "lazy_overhead_pct"):
            if not _is_json_number(summary.get(key)):
                invalid.append(f"summary.{key}: not JSON number")
        if not _is_positive_int(summary.get("population")):
            invalid.append("summary.population: not positive integer")
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
    subspace_candidates: dict[str, dict] = {}
    subspace_seen_candidate_ids: set[str] = set()
    subspace_scored_candidate_ids: set[str] = set()
    subspace_score_keys: set[tuple[object, ...]] = set()
    subspace_selection_rule_hashes: set[str] = set()
    subspace_state_summary_payload: dict | None = None
    subspace_top_k_payload: dict | None = None
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
                        candidate_id = row.get("candidate_id")
                        if isinstance(candidate_id, str) and candidate_id:
                            projected = _candidate_identity_projection(row)
                            if candidate_id in subspace_seen_candidate_ids:
                                invalid.append(f"{rel}: row {line_no} duplicate candidate_id {candidate_id!r}")
                            subspace_seen_candidate_ids.add(candidate_id)
                            subspace_candidates.setdefault(candidate_id, projected)
                    elif rel == "candidates.jsonl":
                        if row.get("scale_mode") not in SUBSPACE_SCALE_MODES:
                            invalid.append(f"{rel}: row {line_no} invalid scale_mode")
                        if row.get("budget_policy") not in SUBSPACE_BUDGET_POLICIES:
                            invalid.append(f"{rel}: row {line_no} invalid budget_policy")
                    if is_subspace_contract and rel == "candidate_scores.jsonl":
                        _check_subspace_score(invalid, f"{rel}: row {line_no}", row, summary)
                        candidate_id = row.get("candidate_id")
                        if isinstance(candidate_id, str) and candidate_id:
                            subspace_scored_candidate_ids.add(candidate_id)
                        if isinstance(row.get("selection_rule_hash"), str) and row.get("selection_rule_hash"):
                            subspace_selection_rule_hashes.add(row["selection_rule_hash"])
                        score_key = (
                            row.get("candidate_id"),
                            row.get("split"),
                            row.get("scorer_name"),
                            row.get("prompt_ids_hash"),
                            row.get("sample_set_hash"),
                            row.get("decode_config_hash"),
                        )
                        if score_key in subspace_score_keys:
                            invalid.append(f"{rel}: row {line_no} duplicate score row for {score_key!r}")
                        subspace_score_keys.add(score_key)
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
    if is_subspace_contract:
        for candidate_id in sorted(subspace_scored_candidate_ids - set(subspace_candidates)):
            invalid.append(f"candidate_scores.jsonl.candidate_id: unknown {candidate_id!r}")
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
        if is_subspace_artifact_contract and rel in SUBSPACE_EXPECTED_SCHEMAS:
            expected_schema = SUBSPACE_EXPECTED_SCHEMAS[rel]
            if payload.get("schema_version") != expected_schema:
                invalid.append(f"{rel}.schema_version: expected {expected_schema!r}, got {payload.get('schema_version')!r}")
        for field in (contract.required_json_nonempty_fields or {}).get(rel, ()):
            value = payload.get(field)
            if not value:
                invalid.append(f"{rel}.{field}: empty")
        for field, expected in (contract.expected_json_values or {}).get(rel, {}).items():
            observed = payload.get(field)
            if observed != expected:
                invalid.append(f"{rel}.{field}: expected {expected!r}, got {observed!r}")
        if is_subspace_contract and rel == "subspace_state_summary.json":
            subspace_state_summary_payload = payload
            _check_subspace_state_summary(invalid, rel, payload)
        if rel == "top_k_ensemble.json" and "candidates" in fields:
            subspace_top_k_payload = payload
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
                    candidate_id = candidate.get("candidate_id")
                    if isinstance(candidate_id, str) and candidate_id:
                        expected = subspace_candidates.get(candidate_id)
                        if expected is None:
                            invalid.append(f"{rel}.candidates[{index}].candidate_id: unknown {candidate_id!r}")
                        elif _candidate_identity_projection(candidate) != expected:
                            invalid.append(f"{rel}.candidates[{index}]: identity differs from candidates.jsonl")
                        if candidate_id not in subspace_scored_candidate_ids:
                            invalid.append(f"{rel}.candidates[{index}].candidate_id: no candidate_scores row")
        if rel == "validation_report.json":
            required_sections = SUBSPACE_VALIDATION_SECTIONS if is_subspace_contract else fields
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
                        evidence_path = _path_under_root(contract.root, evidence)
                        if evidence_path is None:
                            invalid.append(f"{rel}.{section}.evidence_paths: missing or outside run bundle {evidence!r}")
                            continue
                        if evidence_path.name in {"summary.json", "validation_report.json"}:
                            invalid.append(f"{rel}.{section}.evidence_paths: self-attesting {evidence!r}")
                            continue
                        try:
                            evidence_payload = json.loads(evidence_path.read_text())
                        except json.JSONDecodeError:
                            invalid.append(f"{rel}.{section}.evidence_paths: evidence is not JSON {evidence!r}")
                            continue
                        _check_validation_evidence_payload(invalid, rel, section, evidence, evidence_payload)
                failures = value.get("failures")
                if not isinstance(failures, list):
                    invalid.append(f"{rel}.{section}.failures: not list")
                elif failures:
                    invalid.append(f"{rel}.{section}.failures: nonempty")
                if section == "scientific_gate_contract":
                    _check_scientific_gate_contract(
                        invalid,
                        rel,
                        value,
                        root=contract.root,
                        summary=summary,
                        top_k=subspace_top_k_payload,
                        selection_rule_hashes=subspace_selection_rule_hashes,
                    )
        if is_subspace_artifact_contract and rel == "systems_report.json":
            _check_subspace_systems(invalid, rel, payload, contract.root, suite_report=is_subspace_systems_contract)
    if is_subspace_contract:
        if subspace_state_summary_payload is not None:
            _check_subspace_state_payload(invalid, contract.root, summary, subspace_state_summary_payload)
        state_path = contract.root / "subspace_state.pt"
        if state_path.exists() and isinstance(summary.get("subspace_state_hash"), str) and summary.get("subspace_state_hash"):
            if _sha256_path(state_path) != summary["subspace_state_hash"]:
                invalid.append("summary.subspace_state_hash: does not match subspace_state.pt sha256")
        scores_path = contract.root / "candidate_scores.jsonl"
        if scores_path.exists() and isinstance(summary.get("candidate_scores_hash"), str) and summary.get("candidate_scores_hash"):
            if _sha256_path(scores_path) != summary["candidate_scores_hash"]:
                invalid.append("summary.candidate_scores_hash: does not match candidate_scores.jsonl sha256")
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
                    "model_id_or_path",
                    "model_revision",
                    "tokenizer_hash",
                    "task_config_hash",
                    "prompt_contract_hash",
                    "screen_split_hash",
                    "holdout_split_hash",
                    "scorer_name",
                    "scorer_version",
                    "prompt_ids_hash",
                    "sample_set_hash",
                    "prompt_scoring_config_hash",
                    "decode_config_hash",
                    "kernel",
                    "resolved_target_scales",
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
                    "kernel": config.kernel,
                }
                required_jsonl_nonempty = ("candidates.jsonl", "candidate_scores.jsonl")
                required_jsonl_fields = {
                    "candidates.jsonl": SUBSPACE_CANDIDATE_FIELDS,
                    "candidate_scores.jsonl": SUBSPACE_CANDIDATE_SCORE_FIELDS,
                }
                expected_jsonl_counts = {"candidates.jsonl": "population"}
                required_bool_keys = ()
                required_json_fields = {
                    "subspace_state_summary.json": SUBSPACE_STATE_SUMMARY_FIELDS,
                    "top_k_ensemble.json": (*SUBSPACE_JSON_PROVENANCE_FIELDS, *SUBSPACE_TOP_K_FIELDS),
                    "validation_report.json": (*SUBSPACE_JSON_PROVENANCE_FIELDS, *SUBSPACE_VALIDATION_SECTIONS),
                    "systems_report.json": (*SUBSPACE_JSON_PROVENANCE_FIELDS, *SUBSPACE_SYSTEMS_FIELDS, *SUBSPACE_SYSTEMS_AXIS_FIELDS),
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
        systems_csvs = ["subspace_systems.csv"]
        systems_files = ["report.md", "systems_report.json", "subspace_systems.csv"]
        systems_json_fields = {
            "systems_report.json": (
                *SUBSPACE_JSON_PROVENANCE_FIELDS,
                *SUBSPACE_SYSTEMS_FIELDS,
                *SUBSPACE_SYSTEMS_AXIS_FIELDS,
                "source_report",
                "source_run_dir",
            )
        }
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
    add_config_args(parser, include_out=False)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
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
