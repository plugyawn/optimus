from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import subprocess
import sys

import torch

from optimus.evaluation.validation import check_run, gpu_suite_contracts, summary_payload
from optimus.runs.gpu_suite import GpuSuiteConfig, execute_specs, gpu_suite_specs, parse_int_tuple, plan_payload, spec_is_complete


CANDIDATE = "lora:isotropic:seed1:s0.0075:sign1:r8:tq_proj,v_proj"
PNG_1X1 = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02\xfeA\xba\xa3\x8b\x00\x00\x00\x00IEND\xaeB`\x82"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _torch_tensor_sha256(tensor: torch.Tensor) -> str:
    buffer = io.BytesIO()
    torch.save(tensor.detach().cpu().contiguous(), buffer)
    return _sha256_bytes(buffer.getvalue())


def _torch_save_bytes(payload: dict) -> bytes:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def _json_bytes(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True).encode("utf-8")


def _json_sha256(payload: dict) -> str:
    return _sha256_bytes(_json_bytes(payload))


def _summary_for_kind(kind: str, **identity):
    full_search_reference = identity.get("full_search_reference") or identity.get("reference")
    base = {
        "kind": f"vllm_lora_{kind}",
        "method": "lora",
        "model": "Qwen/Qwen3-4B",
        "family": "isotropic",
        "rank": 8,
        "sigma": 0.0075,
        "seed": 2468,
        "targets": ["q_proj", "v_proj"],
        "max_new_tokens": 32,
        "tensor_parallel_size": 1,
        "max_loras": 32,
        "max_cpu_loras": 8192,
        **identity,
    }
    if kind == "bench":
        return base | {
            "adapters": identity.get("adapters", 8),
            "prompts": identity.get("prompts", 64),
            "adapter_build_s": 1.0,
            "load_s": 1.0,
            "lora_tokens_per_sec": None,
            "mixed_tokens_per_sec": 10.0,
            "mixed_prompts_per_sec": 2.0,
        }
    if kind == "search":
        return base | {
            "population": identity.get("population", 1024),
            "screen_prompts": identity.get("screen_prompts", 64),
            "holdout_prompts": identity.get("holdout_prompts", 256),
            "promote": identity.get("promote", 64),
            "chunk_adapters": identity.get("chunk_adapters", 32),
            "antithetic": identity.get("antithetic", True),
            "base_holdout_exact": 0.1,
            "candidate_sec": 1.0,
            "screen_prompts_per_sec": 10.0,
            "screen_tokens_per_sec": 100.0,
            "holdout_tokens_per_sec": 90.0,
            "best_tokens_per_sec": 100.0,
            "eval_elapsed_s": 1.0,
            "load_s": 1.0,
            "top_screen": [{"candidate": CANDIDATE, "exact_mean": 0.2}],
            "top_holdout": [{"candidate": CANDIDATE, "exact_mean": 0.2}],
        }
    if kind == "halving":
        return base | {
            "population": identity.get("population", 1024),
            "screen_prompts": identity.get("screen_prompts", 64),
            "holdout_prompts": identity.get("holdout_prompts", 256),
            "promote": identity.get("promote", 64),
            "chunk_adapters": identity.get("chunk_adapters", 32),
            "antithetic": identity.get("antithetic", True),
            "stage_prompts": identity.get("stage_prompts", 8),
            "survivors": identity.get("survivors", 64),
            "base_holdout_exact": 0.1,
            "candidate_sec": 1.0,
            "stage_candidate_sec": 1.0,
            "screen_candidate_sec": 1.0,
            "prompt_eval_savings": 0.5,
            "eval_elapsed_s": 1.0,
            "top8_survivor_recall": 1.0,
            "top8_possible": 1,
            "full_best_survived": True,
            "halving_selected_regret_vs_full": 0.0,
            "full_search_reference": str(full_search_reference),
            "top_stage": [{"candidate": CANDIDATE, "exact_mean": 0.2}],
            "top_screen": [{"candidate": CANDIDATE, "exact_mean": 0.2}],
            "top_holdout": [{"candidate": CANDIDATE, "exact_mean": 0.2}],
        }
    raise AssertionError(kind)


def write_contract_outputs(contract) -> None:
    contract.root.mkdir(parents=True, exist_ok=True)
    if contract.name == "systems_report":
        csv_rows = {
            "bench.csv": "suite,run,adapters,mixed_tokens_per_sec\noptimus_gpu_suite,bench_a8_p64,8,10\n",
            "full_search.csv": "suite,run,population,candidate_sec\noptimus_gpu_suite,search_p1024_chunk32,1024,1\n",
            "best_of_n.csv": "suite,run,n,best_screen_exact\noptimus_gpu_suite,search_p1024_chunk32,1,0.2\n",
            "quality_scaling.csv": "suite,run,screen_selected_holdout_exact,screen_selected_holdout_delta_vs_base,promoted_holdout_oracle_exact,promoted_holdout_oracle_delta_vs_base\noptimus_gpu_suite,search_p1024_chunk32,0.2,0.1,0.3,0.2\n",
            "parity.csv": "suite,run,trusted_name,candidate_name,n_common,pass,pass_protocol,pass_base_rows,pass_adapter_tensors,pass_output_diff\nbackend_parity_gate,gate,peft,vllm,1,true,true,true,true,true\n",
            "halving.csv": "suite,run,stage_prompts,survivors,prompt_eval_savings\noptimus_gpu_suite,halving_p1024_stage8_surv64,8,64,0.5\n",
            "subspace_systems.csv": "source_run_dir,benchmark_kind,gpu_model,population,target_preset,basis_rank,kernel,candidate_batch_size,candidates_per_sec,top_k_ensemble_cost_multiplier\nrun,subspace,test-gpu,16,transformer-linears,128,torch,4,1.0,1.0\n",
        }
        source_run = contract.root / "source_run"
        source_run.mkdir(parents=True, exist_ok=True)
        (source_run / "timing_trace.jsonl").write_text(json.dumps({"event": "suite_timed_region", "elapsed_s": 0.1, "cuda_synchronized": True}) + "\n")
        source_report = source_run / "systems_report.json"
        source_report.write_text(json.dumps({"schema_version": "subspace_systems_report_v1"}) + "\n")
        for rel in contract.required_files:
            path = contract.root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".png":
                path.write_bytes(PNG_1X1)
            elif path.name == "systems_report.json":
                path.write_text(
                    json.dumps(
                        {
                            "schema_version": "subspace_systems_report_v1",
                            "created_at": "2026-05-25T00:00:00Z",
                            "optimus_version": "0.1.0",
                            "git_commit": "testcommit",
                            "git_dirty": False,
                            "command": ["optimus", "systems-report"],
                            "environment": {"python": "test"},
                            "model_id_or_path": "Qwen/Qwen3-4B",
                            "model_revision": "testrev",
                            "tokenizer_hash": "tok123",
                            "task_config_hash": "task123",
                            "prompt_contract_hash": "promptcontract123",
                            "screen_split_hash": "screen123",
                            "holdout_split_hash": "holdout123",
                            "decode_config_hash": "decode123",
                            "warmup_policy": "one_warmup_batch",
                            "cuda_sync_policy": "sync_timed_regions",
                            "benchmark_kind": "subspace",
                            "population": 16,
                            "target_preset": "transformer-linears",
                            "basis_rank": 128,
                            "kernel": "torch",
                            "candidate_batch_size": 4,
                            "candidate_shard_id": "single",
                            "gpu_model": "test-gpu",
                            "gpu_count": 1,
                            "gpu_memory_allocated_bytes": 1024,
                            "gpu_memory_reserved_bytes": 2048,
                            "base_model_time_s": 1.0,
                            "qx_time_s": 0.05,
                            "lazy_delta_time_s": 0.1,
                            "scoring_time_s": 0.3,
                            "setup_time_s": 0.4,
                            "candidates_per_sec": 1.0,
                            "prompts_per_sec": 2.0,
                            "output_tokens_per_sec": 3.0,
                            "lazy_overhead_pct": 10.0,
                            "prefix_cache_policy": "disabled-for-search",
                            "top_k_ensemble_cost_multiplier": 1.0,
                            "screen_score": 0.1,
                            "holdout_score": 0.2,
                            "screen_to_holdout_drop": -0.1,
                            "diversity_metrics": {"distinct_answers": 1},
                            "random_q_control": {"score": 0.1},
                            "shuffled_q_control": {"score": 0.1},
                            "antithetic_odd_even": {"odd": 0.0, "even": 0.0},
                            "timing_evidence_paths": ["timing_trace.jsonl"],
                            "source_report": str(source_report),
                            "source_run_dir": str(source_run),
                        }
                    )
                    + "\n"
                )
            else:
                path.write_text(csv_rows.get(rel, "placeholder\n"))
        return
    if "summary.json" in contract.required_files:
        kind = "bench" if contract.name.startswith("bench") else "halving" if contract.name.startswith("halving") else "search"
        reference = contract.root.parent / "search_p1024_chunk32"
        if kind == "halving":
            reference.mkdir(parents=True, exist_ok=True)
        summary = _summary_for_kind(kind, full_search_reference=str(reference))
        (contract.root / "summary.json").write_text(json.dumps(summary) + "\n")
    for rel in contract.required_files:
        path = contract.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.name == "summary.json":
            continue
        if path.suffix == ".jsonl":
            if rel == "candidate_summary.jsonl" and kind == "search":
                count = int(summary["population"])
            elif rel == "stage_candidate_summary.jsonl":
                count = int(summary["population"])
            elif rel == "candidate_summary.jsonl" and kind == "halving":
                count = int(summary["survivors"])
            else:
                count = 1
            row = {"candidate": CANDIDATE, "exact_mean": 0.2, "mode": "screen"}
            path.write_text("".join(json.dumps(row) + "\n" for _ in range(count)))
        else:
            path.write_text("placeholder\n")


def write_completed_spec(spec) -> None:
    kind = spec.kind
    if kind == "report":
        return
    spec.output_path.mkdir(parents=True, exist_ok=True)
    (spec.output_path / "summary.json").write_text(json.dumps(_summary_for_kind(kind, **spec.identity)) + "\n")


def write_subspace_contract_outputs(contract) -> None:
    contract.root.mkdir(parents=True, exist_ok=True)
    candidate = {
        "candidate_id": "seed1:+:rho0.01",
        "direction_seed": 1,
        "sign": "+",
        "basis_hash": "basis123",
        "target_set_hash": "target123",
        "scale_mode": "relative-output-rms",
        "rho_or_sigma_w": 0.01,
        "budget_policy": "per-block-equal",
        "budget_hash": "budget123",
        "rng_version": "gaussian_hash_v1",
        "runtime_dtype": "bf16",
        "radius_index": 0,
        "target_preset": "transformer-linears",
        "basis_rank": 128,
        "shard_id": "single",
        "shard_population_start": 0,
        "shard_population_end": 16,
        "worker_id": "worker0",
        "device_id": "cuda:0",
        "prompt_scoring_config_hash": "promptscore123",
    }
    basis_tensor = torch.eye(8, 16)
    subspace_state_bytes = _torch_save_bytes(
        {
            "schema_version": "subspace_state_payload_v1",
            "basis_tensors": {"basis/layer_0.attn_in": basis_tensor},
        }
    )
    candidate_scores_text = "".join(
        json.dumps(
            {
                "candidate_id": f"seed{idx}:+:rho0.01",
                "split": "screen",
                "selection_stage": "screen",
                "selection_rule_hash": "select123",
                "promoted_by_candidate_id": None,
                "scorer_name": "countdown",
                "scorer_version": "countdown_v1",
                "aggregate_metrics": {"exact": 0.1},
                "sample_count": 8,
                "prompt_ids_hash": "prompts123",
                "sample_set_hash": "samples123",
                "decode_config_hash": "decode123",
                "elapsed_s": 0.01,
                "output_tokens": 16,
            }
        )
        + "\n"
        for idx in range(1, 17)
    )
    summary = {
        "schema_version": "subspace_run_summary_v1",
        "kind": "subspace_vllm_search",
        "backend": "vllm",
        "method": "subspace",
        "created_at": "2026-05-25T00:00:00Z",
        "optimus_version": "0.1.0",
        "git_commit": "testcommit",
        "git_dirty": False,
        "command": ["optimus", "search", "--backend", "vllm", "--method", "subspace"],
        "environment": {"python": "test"},
        "model_id_or_path": "Qwen/Qwen3-4B",
        "model_revision": "testrev",
        "tokenizer_hash": "tok123",
        "task_config_hash": "task123",
        "prompt_contract_hash": "promptcontract123",
        "screen_split_hash": "screen123",
        "holdout_split_hash": "holdout123",
        "screen_holdout_overlap": 0,
        "population": 16,
        "basis_hash": "basis123",
        "target_set_hash": "target123",
        "basis_collection_config_hash": "basisconfig123",
        "subspace_state_hash": _sha256_bytes(subspace_state_bytes),
        "candidate_scores_hash": _sha256_bytes(candidate_scores_text.encode("utf-8")),
        "scale_mode": "relative-output-rms",
        "rho_grid": [0.01],
        "sigma_w_grid": None,
        "budget_policy": "per-block-equal",
        "rng_version": "gaussian_hash_v1",
        "candidate_routing": "row_candidate_id",
        "prefix_cache_policy": "disabled-for-search",
        "scorer_name": "countdown",
        "scorer_version": "countdown_v1",
        "prompt_ids_hash": "prompts123",
        "sample_set_hash": "samples123",
        "prompt_scoring_config_hash": "promptscore123",
        "decode_config_hash": "decode123",
        "kernel": "torch",
        "resolved_target_scales": [{"target_id": "layer_0.self_attn.q_proj", "beta_t_by_radius": {"0.01": 0.001}}],
        "candidates_per_sec": 1.0,
        "prompts_per_sec": 2.0,
        "output_tokens_per_sec": 3.0,
        "lazy_overhead_pct": 10.0,
    }
    artifact_provenance = {
        "created_at": summary["created_at"],
        "optimus_version": summary["optimus_version"],
        "git_commit": summary["git_commit"],
        "git_dirty": summary["git_dirty"],
        "command": summary["command"],
        "environment": summary["environment"],
        "model_id_or_path": summary["model_id_or_path"],
        "model_revision": summary["model_revision"],
        "tokenizer_hash": summary["tokenizer_hash"],
        "task_config_hash": summary["task_config_hash"],
        "prompt_contract_hash": summary["prompt_contract_hash"],
        "screen_split_hash": summary["screen_split_hash"],
        "holdout_split_hash": summary["holdout_split_hash"],
        "decode_config_hash": summary["decode_config_hash"],
    }
    validation_sections = [
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
    ]
    validation_report = {
        "schema_version": "validation_report_v1",
        **artifact_provenance,
        **{
            key: {"status": "pass", "evidence_paths": [f"evidence/{key}.json"], "failures": []}
            for key in validation_sections
        },
    }
    json_files = {
        "summary.json": summary,
        "subspace_state_summary.json": {
            "schema_version": "subspace_state_v1",
            **artifact_provenance,
            "model_id_or_path": "Qwen/Qwen3-4B",
            "model_revision": "testrev",
            "tokenizer_hash": "tok123",
            "prompt_contract_hash": "promptcontract123",
            "basis_hash": "basis123",
            "target_preset": "transformer-linears",
            "explicit_targets": [],
            "layers": "all",
            "basis_kind": "activation-svd",
            "basis_centering": "none",
            "basis_token_source": "prefill",
            "basis_split": "train",
            "activation_sites": [
                {
                    "site_id": "layer_0.attn_in",
                    "architecture_family": "qwen3_text",
                    "layer_index": 0,
                    "block_path": "model.layers.0",
                    "read_tensor_path": "model.layers.0.input_layernorm.output",
                    "hook_point": "pre_linear",
                    "norm_position": "post_rmsnorm",
                    "shape_convention": "tokens_hidden",
                    "runtime_dtype": "bf16",
                    "accumulation_dtype": "fp32",
                    "tensor_parallel_sharding_policy": "replicated",
                    "target_module_ids": ["layer_0.self_attn.q_proj"],
                    "calibration_prompt_ids_hash": "prompts123",
                    "calibration_decode_config_hash": "decode123",
                    "basis_control_seed": 123,
                    "transductive": False,
                    "input_dim": 16,
                    "basis_kind": "activation-svd",
                    "requested_rank": 8,
                    "effective_rank": 8,
                    "basis_tensor_key": "basis/layer_0.attn_in",
                    "basis_tensor_sha256": _torch_tensor_sha256(basis_tensor),
                    "singular_values": [1.0, 0.5],
                    "captured_energy": 0.9,
                    "prefill_captured_energy": 0.9,
                    "decode_captured_energy": None,
                    "H_s": 1.0,
                    "A_s": 1.1,
                    "orthonormality_error": 0.0,
                    "gram_error": 0.0,
                    "num_calibration_tokens": 32,
                }
            ],
            "targets": [
                {
                    "target_id": "layer_0.self_attn.q_proj",
                    "activation_site_id": "layer_0.attn_in",
                    "output_dim": 16,
                    "base_output_power_P_t": 1.0,
                }
            ],
        },
        "top_k_ensemble.json": {
            "ensemble_kind": "lazy_top_k",
            "schema_version": "top_k_ensemble_v1",
            **artifact_provenance,
            "aggregation": "majority-vote",
            "tie_break_policy": "lowest_candidate_id",
            "selection_rule": "screen_top_k_fixed_config",
            "K": 1,
            "candidates": [candidate],
            "basis_hash": "basis123",
            "basis_collection_config_hash": "basisconfig123",
            "subspace_state_hash": summary["subspace_state_hash"],
            "scale_mode": "relative-output-rms",
            "rho_or_sigma_w": 0.01,
            "budget_policy": "per-block-equal",
            "target_set_hash": "target123",
            "candidate_scores_hash": summary["candidate_scores_hash"],
            "rng_version": "gaussian_hash_v1",
            "scorer_version": "countdown_v1",
            "prompt_ids_hash": "prompts123",
            "sample_set_hash": "samples123",
            "prompt_scoring_config_hash": "promptscore123",
            "runtime_config_hash": "runtime123",
            "decode_config_hash": "decode123",
        },
        "validation_report.json": validation_report,
        "systems_report.json": {
            "schema_version": "subspace_systems_report_v1",
            **artifact_provenance,
            "warmup_policy": "one_warmup_batch",
            "cuda_sync_policy": "sync_timed_regions",
            "benchmark_kind": "subspace",
            "population": 16,
            "target_preset": "transformer-linears",
            "basis_rank": 128,
            "kernel": "torch",
            "candidate_batch_size": 4,
            "candidate_shard_id": "single",
            "gpu_model": "test-gpu",
            "gpu_count": 1,
            "gpu_memory_allocated_bytes": 1024,
            "gpu_memory_reserved_bytes": 2048,
            "base_model_time_s": 1.0,
            "qx_time_s": 0.05,
            "lazy_delta_time_s": 0.1,
            "scoring_time_s": 0.3,
            "setup_time_s": 0.4,
            "candidates_per_sec": 1.0,
            "prompts_per_sec": 2.0,
            "output_tokens_per_sec": 3.0,
            "lazy_overhead_pct": 10.0,
            "prefix_cache_policy": "disabled-for-search",
            "top_k_ensemble_cost_multiplier": 1.0,
            "screen_score": 0.1,
            "holdout_score": 0.2,
            "screen_to_holdout_drop": -0.1,
            "diversity_metrics": {"distinct_answers": 1},
            "random_q_control": {"score": 0.1},
            "shuffled_q_control": {"score": 0.1},
            "antithetic_odd_even": {"odd": 0.0, "even": 0.0},
            "timing_evidence_paths": ["timing_trace.jsonl"],
        },
    }
    gate_config_artifacts = {}
    observed_configs = []
    for basis_kind, artifact_path in [
        ("activation-svd", "gate/config_activation_svd.json"),
        ("random-orthonormal", "gate/config_random.json"),
        ("shuffled-activation-svd", "gate/config_shuffled.json"),
    ]:
        config_artifact = {
            "schema_version": "scientific_gate_config_v1",
            "basis_kind": basis_kind,
            "K": 1,
            "basis_rank": 128,
            "radius": 0.01,
            "target_preset": "transformer-linears",
            "scale_mode": "relative-output-rms",
            "aggregation": "majority-vote",
            "primary_metric": "top_k_holdout_exact",
            "selection_rule_hash": "select123",
        }
        gate_config_artifacts[artifact_path] = config_artifact
        observed_configs.append(
            {
                "basis_kind": basis_kind,
                "K": 1,
                "basis_rank": 128,
                "radius": 0.01,
                "target_preset": "transformer-linears",
                "scale_mode": "relative-output-rms",
                "aggregation": "majority-vote",
                "artifact_path": artifact_path,
                "artifact_hash": _json_sha256(config_artifact),
            }
        )
    gate_family_artifact = {
        "schema_version": "scientific_gate_family_v1",
        "primary_metric": "top_k_holdout_exact",
        "multiple_comparison_correction": "none_predeclared_single_config",
        "selection_rule_hash": "select123",
        "holdout_tuned": False,
        "K_grid": [1],
        "basis_rank_grid": [128],
        "radius_grid": [0.01],
        "observed_configs": observed_configs,
    }
    random_control_artifact = {
        "schema_version": "scientific_gate_control_v1",
        "basis_kind": "random-orthonormal",
        "metric": "top_k_holdout_exact",
        "sample_set_hash": "samples123",
    }
    shuffled_control_artifact = {
        "schema_version": "scientific_gate_control_v1",
        "basis_kind": "shuffled-activation-svd",
        "metric": "top_k_holdout_exact",
        "sample_set_hash": "samples123",
    }
    random_control_hash = _json_sha256(random_control_artifact)
    shuffled_control_hash = _json_sha256(shuffled_control_artifact)
    random_contrast_artifact = {
        "schema_version": "scientific_gate_contrast_v1",
        "basis_kind": "activation-svd",
        "control_basis_kind": "random-orthonormal",
        "metric": "top_k_holdout_exact",
        "control_artifact_hash": random_control_hash,
        "K": 1,
        "basis_rank": 128,
        "radius": 0.01,
        "target_preset": "transformer-linears",
        "scale_mode": "relative-output-rms",
        "aggregation": "majority-vote",
    }
    shuffled_contrast_artifact = {
        "schema_version": "scientific_gate_contrast_v1",
        "basis_kind": "activation-svd",
        "control_basis_kind": "shuffled-activation-svd",
        "metric": "top_k_holdout_exact",
        "control_artifact_hash": shuffled_control_hash,
        "K": 1,
        "basis_rank": 128,
        "radius": 0.01,
        "target_preset": "transformer-linears",
        "scale_mode": "relative-output-rms",
        "aggregation": "majority-vote",
    }
    gate_artifacts = {
        "gate/gate_family.json": gate_family_artifact,
        **gate_config_artifacts,
        "gate/control_random.json": random_control_artifact,
        "gate/control_shuffled.json": shuffled_control_artifact,
        "gate/contrast_random.json": random_contrast_artifact,
        "gate/contrast_shuffled.json": shuffled_contrast_artifact,
    }
    json_files["validation_report.json"]["scientific_gate_contract"].update(
        {
            "locked_config_hash": "runtime123",
            "selection_rule_hash": "select123",
            "primary_metric": "top_k_holdout_exact",
            "multiple_comparison_correction": "none_predeclared_single_config",
            "K_grid": [1],
            "basis_rank_grid": [128],
            "radius_grid": [0.01],
            "gate_family_artifact_path": "gate/gate_family.json",
            "gate_family_artifact_hash": _json_sha256(gate_family_artifact),
            "compared_control_artifact_paths": {
                "random-orthonormal": "gate/control_random.json",
                "shuffled-activation-svd": "gate/control_shuffled.json",
            },
            "compared_control_artifact_hashes": {
                "random-orthonormal": random_control_hash,
                "shuffled-activation-svd": shuffled_control_hash,
            },
            "tested_contrasts": [
                {
                    "basis_kind": "activation-svd",
                    "control_basis_kind": "random-orthonormal",
                    "metric": "top_k_holdout_exact",
                    "artifact_path": "gate/contrast_random.json",
                    "artifact_hash": _json_sha256(random_contrast_artifact),
                    "control_artifact_path": "gate/control_random.json",
                    "control_artifact_hash": random_control_hash,
                    "K": 1,
                    "basis_rank": 128,
                    "radius": 0.01,
                    "target_preset": "transformer-linears",
                    "scale_mode": "relative-output-rms",
                    "aggregation": "majority-vote",
                },
                {
                    "basis_kind": "activation-svd",
                    "control_basis_kind": "shuffled-activation-svd",
                    "metric": "top_k_holdout_exact",
                    "artifact_path": "gate/contrast_shuffled.json",
                    "artifact_hash": _json_sha256(shuffled_contrast_artifact),
                    "control_artifact_path": "gate/control_shuffled.json",
                    "control_artifact_hash": shuffled_control_hash,
                    "K": 1,
                    "basis_rank": 128,
                    "radius": 0.01,
                    "target_preset": "transformer-linears",
                    "scale_mode": "relative-output-rms",
                    "aggregation": "majority-vote",
                },
            ],
            "locked_K": 1,
            "locked_basis_rank": 128,
            "locked_radius": 0.01,
            "locked_target_preset": "transformer-linears",
            "locked_scale_mode": "relative-output-rms",
            "locked_aggregation": "majority-vote",
            "selection_split": "screen",
            "holdout_tuned": False,
            "screen_holdout_overlap": 0,
            "gate_stage": "production",
            "basis_kind": "activation-svd",
            "control_basis_kinds": ["random-orthonormal", "shuffled-activation-svd"],
            "comparison": "activation_svd_minus_best_control",
            "gate_type": "paired-bootstrap-positive",
            "epsilon": 0.0,
            "confidence_interval": {"lower": 0.01, "upper": 0.1},
        }
    )
    for rel in contract.required_files:
        path = contract.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if rel == "subspace_state.pt":
            path.write_bytes(subspace_state_bytes)
        elif rel == "candidates.jsonl":
            path.write_text("".join(json.dumps(candidate | {"candidate_id": f"seed{idx}:+:rho0.01", "direction_seed": idx}) + "\n" for idx in range(1, 17)))
        elif rel == "candidate_scores.jsonl":
            path.write_text(candidate_scores_text)
        elif rel in json_files:
            path.write_text(json.dumps(json_files[rel]) + "\n")
        else:
            path.write_text("placeholder\n")
    for rel, payload in gate_artifacts.items():
        path = contract.root / rel
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(_json_bytes(payload))
    evidence_dir = contract.root / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    for section in validation_sections:
        payload = {
            "evidence_schema_version": "validation_evidence_v1",
            "section": section,
            "status": "pass",
            "generated_at": "2026-05-25T00:00:00Z",
            "command": ["pytest", section],
            "checks": [{"name": f"{section}_check", "passed": True}],
        }
        if section == "drift_diagnostics":
            payload.update(
                {
                    "probe_split_hash": "drift-probe123",
                    "reference_artifact_hash": "base-logits123",
                    "candidate_artifact_hash": "candidate-logits123",
                    "aggregation": "mean_token_rows",
                    "sample_count": 32,
                    "temperature": 1.0,
                    "epsilon": 1e-6,
                    "metrics": {"logit_kl_mean": 0.01, "hidden_state_rms_drift": 0.02},
                }
            )
        (evidence_dir / f"{section}.json").write_text(
            json.dumps(payload)
            + "\n"
        )
    (contract.root / "timing_trace.jsonl").write_text(json.dumps({"event": "timed_region", "elapsed_s": 0.1, "cuda_synchronized": True}) + "\n")


def test_gpu_suite_specs_include_p1024_and_p4096_searches(tmp_path: Path):
    config = GpuSuiteConfig(output_root=tmp_path / "runs", systems_output_root=tmp_path / "systems")

    specs = gpu_suite_specs(config)
    names = {spec.name for spec in specs}

    assert "search_p1024_chunk32" in names
    assert "search_p4096_chunk32" in names
    assert "halving_p1024_stage8_surv64" not in names
    assert "systems_report" in names


def test_plan_payload_serializes_commands(tmp_path: Path):
    config = GpuSuiteConfig(output_root=tmp_path / "runs", systems_output_root=tmp_path / "systems")

    payload = plan_payload(config)
    search = next(run for run in payload["runs"] if run["name"] == "search_p4096_chunk32")

    assert search["kind"] == "search"
    assert search["command"][:6] == ["optimus", "search", "--backend", "vllm", "--method", "lora"]
    assert "--population" in search["command"]
    assert "4096" in search["command"]
    assert "--tensor-parallel-size" in search["command"]
    assert "1" in search["command"]


def test_plan_payload_respects_full_config_surface(tmp_path: Path):
    config = GpuSuiteConfig(
        output_root=tmp_path / "runs",
        systems_output_root=tmp_path / "systems",
        populations=(128,),
        rank=16,
        sigma=0.0125,
        seed=999,
        targets="q_proj,k_proj,v_proj,o_proj",
        max_new_tokens=48,
        prompt_variants="tight,compact",
        use_chat_template=True,
        chunk_adapters=4,
        max_loras=4,
        max_cpu_loras=2048,
        tensor_parallel_size=2,
        bench_adapters=(4,),
        run_halving=False,
    )

    payload = plan_payload(config)
    names = {run["name"] for run in payload["runs"]}
    search = next(run for run in payload["runs"] if run["kind"] == "search")

    assert "search_p128_chunk4" in names
    assert "halving_p1024_stage8_surv64" not in names
    assert "--rank" in search["command"]
    assert "16" in search["command"]
    assert "--targets" in search["command"]
    assert "q_proj,k_proj,v_proj,o_proj" in search["command"]
    assert "--prompt-variants" in search["command"]
    assert "tight,compact" in search["command"]
    assert "--use-chat-template" in search["command"]


def test_plan_payload_can_keep_search_adapters_for_external_eval(tmp_path: Path):
    config = GpuSuiteConfig(
        output_root=tmp_path / "runs",
        systems_output_root=tmp_path / "systems",
        populations=(128,),
        keep_adapters=True,
        run_halving=False,
    )

    search = next(run for run in plan_payload(config)["runs"] if run["kind"] == "search")

    assert "--keep-adapters" in search["command"]


def test_subspace_plan_uses_final_public_surface(tmp_path: Path):
    config = GpuSuiteConfig(
        output_root=tmp_path / "runs",
        systems_output_root=tmp_path / "systems",
        method="subspace",
        populations=(128,),
        basis_rank=256,
        basis_prompts=64,
        target_preset="transformer-linears",
        rho_grid="0.002,0.005",
    )

    payload = plan_payload(config)
    search = next(run for run in payload["runs"] if run["kind"] == "search")
    commands = [" ".join(run["command"]) for run in payload["runs"]]

    assert search["command"][:6] == ["optimus", "search", "--backend", "vllm", "--method", "subspace"]
    assert search["name"] == "search_p128_subspace_r256"
    assert search["planned_fail_closed"] is True
    assert "--basis-rank" in search["command"]
    assert "256" in search["command"]
    assert "--basis-prompts" in search["command"]
    assert "--target-preset" in search["command"]
    assert "--layers" in search["command"]
    assert "--basis-centering" in search["command"]
    assert "--basis-token-source" in search["command"]
    assert "--candidate-batch-size" in search["command"]
    assert "--kernel" in search["command"]
    assert "--prefix-cache-policy" in search["command"]
    assert "disabled-for-search" in search["command"]
    assert "--rho-grid" in search["command"]
    assert "--rank" not in search["command"]
    assert "--sigma" not in search["command"]
    assert "--max-loras" not in search["command"]
    assert not any("vllm-search" in command or "vllm-bench" in command or "vllm-halving" in command for command in commands)
    for adapter_key in ["rank", "sigma", "targets", "chunk_adapters", "max_loras", "max_cpu_loras", "keep_adapters", "bench_adapters"]:
        assert adapter_key not in payload["config"]


def test_subspace_run_plan_cli_rejects_explicit_lora_only_options(tmp_path: Path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "optimus.cli",
            "run-plan",
            "--method",
            "subspace",
            "--root",
            str(tmp_path / "runs"),
            "--systems-out",
            str(tmp_path / "systems"),
            "--rank",
            "8",
            "--sigma",
            "0.01",
            "--targets",
            "q_proj",
            "--max-loras",
            "4",
            "--bench-adapters",
            "4",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "does not accept LoRA-only options" in result.stderr
    assert "--rank" in result.stderr
    assert "--bench-adapters" in result.stderr


def test_subspace_run_suite_cli_rejects_explicit_lora_only_options(tmp_path: Path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "optimus.cli",
            "run-suite",
            "--dry-run",
            "--no-ensure-data",
            "--method",
            "subspace",
            "--root",
            str(tmp_path / "runs"),
            "--systems-out",
            str(tmp_path / "systems"),
            "--chunk-adapters",
            "4",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "does not accept LoRA-only options" in result.stderr
    assert "--chunk-adapters" in result.stderr


def test_subspace_run_suite_fails_closed_before_execution(tmp_path: Path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "optimus.cli",
            "run-suite",
            "--no-ensure-data",
            "--method",
            "subspace",
            "--root",
            str(tmp_path / "runs"),
            "--systems-out",
            str(tmp_path / "systems"),
            "--populations",
            "16",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "planned fail-closed" in result.stderr


def test_subspace_plan_accepts_documented_screen_matching_flags(tmp_path: Path):
    config = GpuSuiteConfig(
        output_root=tmp_path / "runs",
        systems_output_root=tmp_path / "systems",
        method="subspace",
        populations=(128,),
        match_screen_to_holdout_base_exact=True,
        screen_pool_prompts=512,
    )

    command = next(run for run in plan_payload(config)["runs"] if run["kind"] == "search")["command"]

    assert "--match-screen-to-holdout-base-exact" in command
    assert "--screen-pool-prompts" in command
    assert "512" in command


def test_subspace_plan_rejects_shared_prefix_caching(tmp_path: Path):
    config = GpuSuiteConfig(
        output_root=tmp_path / "runs",
        systems_output_root=tmp_path / "systems",
        method="subspace",
        populations=(128,),
        enable_prefix_caching=True,
    )

    try:
        plan_payload(config)
    except RuntimeError as exc:
        assert "shared prefix caching is forbidden" in str(exc)
    else:
        raise AssertionError("subspace plan must reject shared prefix caching")


def test_halving_plan_uses_full_search_reference_for_regret_metrics(tmp_path: Path):
    config = GpuSuiteConfig(output_root=tmp_path / "runs", systems_output_root=tmp_path / "systems", run_halving=True)

    try:
        plan_payload(config)
    except RuntimeError as exc:
        assert "staged search is disabled" in str(exc)
    else:
        raise AssertionError("run-plan must not emit removed vllm-halving commands")


def test_run_contract_checks_missing_and_present_files(tmp_path: Path):
    config = GpuSuiteConfig(output_root=tmp_path / "runs", systems_output_root=tmp_path / "systems")
    contract = next(item for item in gpu_suite_contracts(config) if item.name == "search_p1024_chunk32")

    initial = check_run(contract)
    assert not initial.passed
    assert "summary.json" in initial.missing

    write_contract_outputs(contract)

    final = check_run(contract)
    assert final.passed
    assert summary_payload([final])["pass"] is True


def test_subspace_contract_requires_subspace_artifacts_and_candidate_identities(tmp_path: Path):
    config = GpuSuiteConfig(
        output_root=tmp_path / "runs",
        systems_output_root=tmp_path / "systems",
        method="subspace",
        populations=(16,),
    )
    contract = next(item for item in gpu_suite_contracts(config) if item.name == "search_p16_subspace_r128")

    write_subspace_contract_outputs(contract)
    passed = check_run(contract)
    assert passed.passed

    (contract.root / "top_k_ensemble.json").write_text(
        json.dumps(
            {
                "ensemble_kind": "lazy_top_k",
                "schema_version": "top_k_ensemble_v1",
                "aggregation": "majority-vote",
                "tie_break_policy": "lowest_candidate_id",
                "selection_rule": "screen_top_k_fixed_config",
                "K": 1,
                "candidate_ids": ["seed1:+:rho0.01"],
                "basis_hash": "basis123",
                "target_set_hash": "target123",
                "scorer_version": "countdown_v1",
                "prompt_ids_hash": "prompts123",
                "runtime_config_hash": "runtime123",
                "decode_config_hash": "decode123",
            }
        )
        + "\n"
    )

    failed = check_run(contract)
    assert not failed.passed
    assert any("top_k_ensemble.json.candidates" in item for item in failed.invalid)


def test_subspace_contract_rejects_bad_routing_cache_and_hollow_reports(tmp_path: Path):
    config = GpuSuiteConfig(
        output_root=tmp_path / "runs",
        systems_output_root=tmp_path / "systems",
        method="subspace",
        populations=(16,),
    )
    contract = next(item for item in gpu_suite_contracts(config) if item.name == "search_p16_subspace_r128")
    write_subspace_contract_outputs(contract)
    summary = json.loads((contract.root / "summary.json").read_text())
    summary["candidate_routing"] = "physical_row_order"
    summary["prefix_cache_policy"] = "enabled_shared"
    (contract.root / "summary.json").write_text(json.dumps(summary) + "\n")
    systems = json.loads((contract.root / "systems_report.json").read_text())
    systems["prefix_cache_policy"] = "enabled_shared"
    (contract.root / "systems_report.json").write_text(json.dumps(systems) + "\n")
    (contract.root / "validation_report.json").write_text(
        json.dumps({key: {} for key in ["math_tests", "rng_replay_tests", "routing_cache_tests", "selector_quality", "holdout_quality", "ensemble_quality", "drift_diagnostics", "random_shuffled_controls", "throughput_gates"]})
        + "\n"
    )

    result = check_run(contract)

    assert not result.passed
    assert any("summary.candidate_routing" in item for item in result.invalid)
    assert any("summary.prefix_cache_policy" in item for item in result.invalid)
    assert any("systems_report.json.prefix_cache_policy" in item for item in result.invalid)
    assert any("validation_report.json.math_tests: empty" in item for item in result.invalid)


def test_subspace_contract_rejects_invalid_top_k_semantics(tmp_path: Path):
    config = GpuSuiteConfig(
        output_root=tmp_path / "runs",
        systems_output_root=tmp_path / "systems",
        method="subspace",
        populations=(16,),
    )
    contract = next(item for item in gpu_suite_contracts(config) if item.name == "search_p16_subspace_r128")
    write_subspace_contract_outputs(contract)
    top_k = json.loads((contract.root / "top_k_ensemble.json").read_text())
    top_k["aggregation"] = "custom-bad-value"
    top_k["K"] = 16
    top_k["candidates"][0].pop("budget_hash")
    (contract.root / "top_k_ensemble.json").write_text(json.dumps(top_k) + "\n")

    result = check_run(contract)

    assert not result.passed
    assert any("top_k_ensemble.json.aggregation" in item for item in result.invalid)
    assert any("top_k_ensemble.json.K" in item for item in result.invalid)
    assert any("top_k_ensemble.json.candidates[1].budget_hash" in item for item in result.invalid)


def test_subspace_contract_rejects_replay_identity_gaps(tmp_path: Path):
    config = GpuSuiteConfig(
        output_root=tmp_path / "runs",
        systems_output_root=tmp_path / "systems",
        method="subspace",
        populations=(16,),
    )
    contract = next(item for item in gpu_suite_contracts(config) if item.name == "search_p16_subspace_r128")
    write_subspace_contract_outputs(contract)
    top_k = json.loads((contract.root / "top_k_ensemble.json").read_text())
    top_k.pop("subspace_state_hash")
    top_k.pop("candidate_scores_hash")
    top_k["basis_collection_config_hash"] = "wrong"
    top_k["candidates"][0]["sign"] = "x"
    top_k["candidates"][0]["basis_hash"] = "wrong"
    (contract.root / "top_k_ensemble.json").write_text(json.dumps(top_k) + "\n")

    result = check_run(contract)

    assert not result.passed
    assert any("top_k_ensemble.json.subspace_state_hash" in item for item in result.invalid)
    assert any("top_k_ensemble.json.candidate_scores_hash" in item for item in result.invalid)
    assert any("top_k_ensemble.json.basis_collection_config_hash" in item for item in result.invalid)
    assert any("top_k_ensemble.json.candidates[1].sign" in item for item in result.invalid)
    assert any("top_k_ensemble.json.candidates[1].basis_hash" in item for item in result.invalid)


def test_subspace_contract_rejects_bad_schema_versions(tmp_path: Path):
    config = GpuSuiteConfig(
        output_root=tmp_path / "runs",
        systems_output_root=tmp_path / "systems",
        method="subspace",
        populations=(16,),
    )
    contract = next(item for item in gpu_suite_contracts(config) if item.name == "search_p16_subspace_r128")
    write_subspace_contract_outputs(contract)
    for rel in ["summary.json", "subspace_state_summary.json", "top_k_ensemble.json", "validation_report.json", "systems_report.json"]:
        payload = json.loads((contract.root / rel).read_text())
        payload["schema_version"] = "bogus_schema_v999"
        (contract.root / rel).write_text(json.dumps(payload) + "\n")

    result = check_run(contract)

    assert not result.passed
    assert any("summary.schema_version" in item for item in result.invalid)
    assert any("top_k_ensemble.json.schema_version" in item for item in result.invalid)
    assert any("validation_report.json.schema_version" in item for item in result.invalid)


def test_subspace_contract_rejects_missing_site_basis_kind(tmp_path: Path):
    config = GpuSuiteConfig(
        output_root=tmp_path / "runs",
        systems_output_root=tmp_path / "systems",
        method="subspace",
        populations=(16,),
    )
    contract = next(item for item in gpu_suite_contracts(config) if item.name == "search_p16_subspace_r128")
    write_subspace_contract_outputs(contract)
    state_summary = json.loads((contract.root / "subspace_state_summary.json").read_text())
    state_summary["activation_sites"][0].pop("basis_kind")
    (contract.root / "subspace_state_summary.json").write_text(json.dumps(state_summary) + "\n")

    result = check_run(contract)

    assert not result.passed
    assert any("subspace_state_summary.json.activation_sites[1].basis_kind: missing" in item for item in result.invalid)


def test_subspace_contract_rejects_state_and_score_hash_drift(tmp_path: Path):
    config = GpuSuiteConfig(
        output_root=tmp_path / "runs",
        systems_output_root=tmp_path / "systems",
        method="subspace",
        populations=(16,),
    )
    contract = next(item for item in gpu_suite_contracts(config) if item.name == "search_p16_subspace_r128")
    write_subspace_contract_outputs(contract)
    (contract.root / "subspace_state.pt").write_bytes(b"not-a-loadable-state")
    (contract.root / "candidate_scores.jsonl").write_text((contract.root / "candidate_scores.jsonl").read_text() + "\n")

    result = check_run(contract)

    assert not result.passed
    assert any("subspace_state_hash" in item for item in result.invalid)
    assert any("subspace_state.pt: cannot load payload" in item for item in result.invalid)
    assert any("candidate_scores_hash" in item for item in result.invalid)


def test_subspace_contract_rejects_self_attesting_validation_evidence(tmp_path: Path):
    config = GpuSuiteConfig(
        output_root=tmp_path / "runs",
        systems_output_root=tmp_path / "systems",
        method="subspace",
        populations=(16,),
    )
    contract = next(item for item in gpu_suite_contracts(config) if item.name == "search_p16_subspace_r128")
    write_subspace_contract_outputs(contract)
    report = json.loads((contract.root / "validation_report.json").read_text())
    report["math_tests"]["evidence_paths"] = ["summary.json"]
    (contract.root / "validation_report.json").write_text(json.dumps(report) + "\n")

    result = check_run(contract)

    assert not result.passed
    assert any("math_tests.evidence_paths: self-attesting" in item for item in result.invalid)


def test_subspace_contract_rejects_validation_evidence_outside_run_bundle(tmp_path: Path):
    config = GpuSuiteConfig(
        output_root=tmp_path / "runs",
        systems_output_root=tmp_path / "systems",
        method="subspace",
        populations=(16,),
    )
    contract = next(item for item in gpu_suite_contracts(config) if item.name == "search_p16_subspace_r128")
    write_subspace_contract_outputs(contract)
    external = tmp_path / "external_evidence.json"
    external.write_text(
        json.dumps(
            {
                "evidence_schema_version": "validation_evidence_v1",
                "section": "math_tests",
                "status": "pass",
                "generated_at": "2026-05-25T00:00:00Z",
                "command": ["pytest", "math_tests"],
                "checks": [{"name": "external", "passed": True}],
            }
        )
        + "\n"
    )
    report = json.loads((contract.root / "validation_report.json").read_text())
    report["math_tests"]["evidence_paths"] = [str(external)]
    (contract.root / "validation_report.json").write_text(json.dumps(report) + "\n")

    result = check_run(contract)

    assert not result.passed
    assert any("math_tests.evidence_paths: missing or outside run bundle" in item for item in result.invalid)


def test_subspace_contract_rejects_hollow_validation_evidence(tmp_path: Path):
    config = GpuSuiteConfig(
        output_root=tmp_path / "runs",
        systems_output_root=tmp_path / "systems",
        method="subspace",
        populations=(16,),
    )
    contract = next(item for item in gpu_suite_contracts(config) if item.name == "search_p16_subspace_r128")
    write_subspace_contract_outputs(contract)
    (contract.root / "evidence" / "math_tests.json").write_text(json.dumps({"section": "math_tests", "status": "pass"}) + "\n")

    result = check_run(contract)

    assert not result.passed
    assert any("math_tests.evidence_paths: evidence_schema_version" in item for item in result.invalid)
    assert any("math_tests.evidence_paths: no substantive evidence payload" in item for item in result.invalid)


def test_subspace_contract_rejects_validation_evidence_without_command(tmp_path: Path):
    config = GpuSuiteConfig(
        output_root=tmp_path / "runs",
        systems_output_root=tmp_path / "systems",
        method="subspace",
        populations=(16,),
    )
    contract = next(item for item in gpu_suite_contracts(config) if item.name == "search_p16_subspace_r128")
    write_subspace_contract_outputs(contract)
    evidence = json.loads((contract.root / "evidence" / "math_tests.json").read_text())
    evidence.pop("command")
    (contract.root / "evidence" / "math_tests.json").write_text(json.dumps(evidence) + "\n")

    result = check_run(contract)

    assert not result.passed
    assert any("math_tests.evidence_paths: command missing" in item for item in result.invalid)


def test_subspace_contract_rejects_incomplete_drift_evidence(tmp_path: Path):
    config = GpuSuiteConfig(
        output_root=tmp_path / "runs",
        systems_output_root=tmp_path / "systems",
        method="subspace",
        populations=(16,),
    )
    contract = next(item for item in gpu_suite_contracts(config) if item.name == "search_p16_subspace_r128")
    write_subspace_contract_outputs(contract)
    evidence = json.loads((contract.root / "evidence" / "drift_diagnostics.json").read_text())
    evidence.pop("candidate_artifact_hash")
    evidence.pop("sample_count")
    evidence.pop("temperature")
    evidence["metrics"].pop("hidden_state_rms_drift")
    (contract.root / "evidence" / "drift_diagnostics.json").write_text(json.dumps(evidence) + "\n")

    result = check_run(contract)

    assert not result.passed
    assert any("drift_diagnostics.evidence_paths: candidate_artifact_hash" in item for item in result.invalid)
    assert any("drift_diagnostics.evidence_paths: sample_count" in item for item in result.invalid)
    assert any("drift_diagnostics.evidence_paths: temperature" in item for item in result.invalid)
    assert any("drift_diagnostics.evidence_paths: metrics.hidden_state_rms_drift" in item for item in result.invalid)


def test_subspace_contract_rejects_holdout_overlap_claims(tmp_path: Path):
    config = GpuSuiteConfig(
        output_root=tmp_path / "runs",
        systems_output_root=tmp_path / "systems",
        method="subspace",
        populations=(16,),
    )
    contract = next(item for item in gpu_suite_contracts(config) if item.name == "search_p16_subspace_r128")
    write_subspace_contract_outputs(contract)
    summary = json.loads((contract.root / "summary.json").read_text())
    summary["screen_holdout_overlap"] = 1
    (contract.root / "summary.json").write_text(json.dumps(summary) + "\n")
    report = json.loads((contract.root / "validation_report.json").read_text())
    report["scientific_gate_contract"]["screen_holdout_overlap"] = 1
    (contract.root / "validation_report.json").write_text(json.dumps(report) + "\n")

    result = check_run(contract)

    assert not result.passed
    assert any("summary.screen_holdout_overlap" in item for item in result.invalid)
    assert any("scientific_gate_contract.screen_holdout_overlap" in item for item in result.invalid)


def test_subspace_contract_rejects_bool_overlap_and_string_summary_numbers(tmp_path: Path):
    config = GpuSuiteConfig(
        output_root=tmp_path / "runs",
        systems_output_root=tmp_path / "systems",
        method="subspace",
        populations=(16,),
    )
    contract = next(item for item in gpu_suite_contracts(config) if item.name == "search_p16_subspace_r128")
    write_subspace_contract_outputs(contract)
    summary = json.loads((contract.root / "summary.json").read_text())
    summary["screen_holdout_overlap"] = False
    summary["candidates_per_sec"] = "1.0"
    summary["prompts_per_sec"] = "2.0"
    summary["output_tokens_per_sec"] = "3.0"
    summary["lazy_overhead_pct"] = "10.0"
    (contract.root / "summary.json").write_text(json.dumps(summary) + "\n")
    report = json.loads((contract.root / "validation_report.json").read_text())
    report["scientific_gate_contract"]["screen_holdout_overlap"] = False
    (contract.root / "validation_report.json").write_text(json.dumps(report) + "\n")

    result = check_run(contract)

    assert not result.passed
    assert any("summary.screen_holdout_overlap" in item for item in result.invalid)
    assert any("summary.candidates_per_sec: not JSON number" in item for item in result.invalid)
    assert any("summary.prompts_per_sec: not JSON number" in item for item in result.invalid)
    assert any("summary.output_tokens_per_sec: not JSON number" in item for item in result.invalid)
    assert any("summary.lazy_overhead_pct: not JSON number" in item for item in result.invalid)
    assert any("scientific_gate_contract.screen_holdout_overlap" in item for item in result.invalid)


def test_subspace_contract_rejects_weak_engineering_exception(tmp_path: Path):
    config = GpuSuiteConfig(
        output_root=tmp_path / "runs",
        systems_output_root=tmp_path / "systems",
        method="subspace",
        populations=(16,),
    )
    contract = next(item for item in gpu_suite_contracts(config) if item.name == "search_p16_subspace_r128")
    write_subspace_contract_outputs(contract)
    report = json.loads((contract.root / "validation_report.json").read_text())
    gate = report["scientific_gate_contract"]
    gate["gate_type"] = "engineering-proceed-no-scientific-win"
    gate["confidence_interval"]["lower"] = -0.01
    gate["engineering_exception"] = {
        "accepted_label": "engineering_proceed_no_scientific_win",
        "operational_advantage": {
            "metric": "logit_kl_mean_reduction_pct",
            "delta": 3.0,
        },
    }
    (contract.root / "validation_report.json").write_text(json.dumps(report) + "\n")

    result = check_run(contract)

    assert not result.passed
    assert any("engineering proceed requires lower >= -epsilon" in item for item in result.invalid)
    assert any("operational_advantage.delta" in item for item in result.invalid)
    assert any("operational_advantage.probe_split_hash" in item for item in result.invalid)
    assert any("operational_advantage.reference_artifact_hash" in item for item in result.invalid)


def test_subspace_contract_accepts_tie_like_engineering_exception(tmp_path: Path):
    config = GpuSuiteConfig(
        output_root=tmp_path / "runs",
        systems_output_root=tmp_path / "systems",
        method="subspace",
        populations=(16,),
    )
    contract = next(item for item in gpu_suite_contracts(config) if item.name == "search_p16_subspace_r128")
    write_subspace_contract_outputs(contract)
    report = json.loads((contract.root / "validation_report.json").read_text())
    gate = report["scientific_gate_contract"]
    gate["gate_type"] = "engineering-proceed-no-scientific-win"
    gate["confidence_interval"] = {"lower": 0.0, "upper": 0.02}
    gate["engineering_exception"] = {
        "accepted_label": "engineering_proceed_no_scientific_win",
        "operational_advantage": {
            "metric": "logit_kl_mean_reduction_pct",
            "delta": 30.0,
            "probe_split_hash": "drift-probe123",
            "reference_artifact_hash": "base-logits123",
            "aggregation": "mean_token_rows",
            "direction": "lower_is_better",
        },
    }
    (contract.root / "validation_report.json").write_text(json.dumps(report) + "\n")

    result = check_run(contract)

    assert result.passed


def test_subspace_contract_rejects_scientific_gate_not_tied_to_artifacts(tmp_path: Path):
    config = GpuSuiteConfig(
        output_root=tmp_path / "runs",
        systems_output_root=tmp_path / "systems",
        method="subspace",
        populations=(16,),
    )
    contract = next(item for item in gpu_suite_contracts(config) if item.name == "search_p16_subspace_r128")
    write_subspace_contract_outputs(contract)
    report = json.loads((contract.root / "validation_report.json").read_text())
    report["scientific_gate_contract"]["locked_config_hash"] = "wrong"
    report["scientific_gate_contract"]["selection_rule_hash"] = "unknown"
    report["scientific_gate_contract"]["locked_K"] = 2
    report["scientific_gate_contract"]["holdout_tuned"] = True
    report["scientific_gate_contract"].pop("compared_control_artifact_hashes")
    report["scientific_gate_contract"].pop("tested_contrasts")
    report["scientific_gate_contract"]["K_grid"] = []
    (contract.root / "validation_report.json").write_text(json.dumps(report) + "\n")

    result = check_run(contract)

    assert not result.passed
    assert any("selection_rule_hash: not present" in item for item in result.invalid)
    assert any("locked_config_hash" in item for item in result.invalid)
    assert any("locked_K" in item for item in result.invalid)
    assert any("holdout_tuned" in item for item in result.invalid)
    assert any("compared_control_artifact_hashes" in item for item in result.invalid)
    assert any("tested_contrasts" in item for item in result.invalid)
    assert any("K_grid" in item for item in result.invalid)


def test_subspace_contract_rejects_scientific_gate_grid_drift_without_correction(tmp_path: Path):
    config = GpuSuiteConfig(
        output_root=tmp_path / "runs",
        systems_output_root=tmp_path / "systems",
        method="subspace",
        populations=(16,),
    )
    contract = next(item for item in gpu_suite_contracts(config) if item.name == "search_p16_subspace_r128")
    write_subspace_contract_outputs(contract)
    report = json.loads((contract.root / "validation_report.json").read_text())
    gate = report["scientific_gate_contract"]
    gate["K_grid"] = [1, 4]
    gate["basis_rank_grid"] = [64]
    gate["radius_grid"] = [0.02]
    gate["multiple_comparison_correction"] = "none_predeclared_single_config"
    (contract.root / "validation_report.json").write_text(json.dumps(report) + "\n")

    result = check_run(contract)

    assert not result.passed
    assert any("locked_basis_rank: not present in basis_rank_grid" in item for item in result.invalid)
    assert any("locked_radius: not present in radius_grid" in item for item in result.invalid)
    assert any("none_predeclared_single_config requires singleton grids" in item for item in result.invalid)


def test_subspace_contract_rejects_scientific_gate_contrast_mismatch(tmp_path: Path):
    config = GpuSuiteConfig(
        output_root=tmp_path / "runs",
        systems_output_root=tmp_path / "systems",
        method="subspace",
        populations=(16,),
    )
    contract = next(item for item in gpu_suite_contracts(config) if item.name == "search_p16_subspace_r128")
    write_subspace_contract_outputs(contract)
    report = json.loads((contract.root / "validation_report.json").read_text())
    contrasts = report["scientific_gate_contract"]["tested_contrasts"]
    contrasts[0]["metric"] = "screen_only_metric"
    contrasts[1]["control_artifact_hash"] = "wrong-control-hash"
    (contract.root / "validation_report.json").write_text(json.dumps(report) + "\n")

    result = check_run(contract)

    assert not result.passed
    assert any("tested_contrasts[1].metric: does not match primary_metric" in item for item in result.invalid)
    assert any("tested_contrasts[2].control_artifact_hash" in item for item in result.invalid)
    assert any("missing primary_metric controls ['random-orthonormal']" in item for item in result.invalid)


def test_subspace_contract_rejects_unbound_scientific_gate_artifacts(tmp_path: Path):
    config = GpuSuiteConfig(
        output_root=tmp_path / "runs",
        systems_output_root=tmp_path / "systems",
        method="subspace",
        populations=(16,),
    )
    contract = next(item for item in gpu_suite_contracts(config) if item.name == "search_p16_subspace_r128")
    write_subspace_contract_outputs(contract)
    report = json.loads((contract.root / "validation_report.json").read_text())
    gate = report["scientific_gate_contract"]
    gate["gate_family_artifact_hash"] = "wrong-gate-hash"
    gate["compared_control_artifact_paths"]["random-orthonormal"] = str(tmp_path / "outside_control.json")
    gate["tested_contrasts"][0]["artifact_path"] = "gate/missing_contrast.json"
    (contract.root / "validation_report.json").write_text(json.dumps(report) + "\n")

    result = check_run(contract)

    assert not result.passed
    assert any("gate_family_artifact_hash: does not match artifact" in item for item in result.invalid)
    assert any("compared_control_artifact_paths.random-orthonormal_path: missing or outside run bundle" in item for item in result.invalid)
    assert any("tested_contrasts[1].artifact_path: missing or outside run bundle" in item for item in result.invalid)


def test_subspace_contract_rejects_validation_split_aliasing_screen(tmp_path: Path):
    config = GpuSuiteConfig(
        output_root=tmp_path / "runs",
        systems_output_root=tmp_path / "systems",
        method="subspace",
        populations=(16,),
    )
    contract = next(item for item in gpu_suite_contracts(config) if item.name == "search_p16_subspace_r128")
    write_subspace_contract_outputs(contract)
    validation_artifact = {
        "schema_version": "validation_selection_artifact_v1",
        "selection_split_hash": "screen123",
        "selection_rule_hash": "select123",
    }
    validation_path = contract.root / "gate" / "validation_selection.json"
    validation_path.write_bytes(_json_bytes(validation_artifact))
    report = json.loads((contract.root / "validation_report.json").read_text())
    gate = report["scientific_gate_contract"]
    gate["multiple_comparison_correction"] = "separate_validation_split"
    gate["validation_selection_split_hash"] = "screen123"
    gate["validation_selection_artifact_path"] = "gate/validation_selection.json"
    gate["validation_selection_artifact_hash"] = _json_sha256(validation_artifact)
    family = json.loads((contract.root / gate["gate_family_artifact_path"]).read_text())
    family["multiple_comparison_correction"] = "separate_validation_split"
    (contract.root / gate["gate_family_artifact_path"]).write_bytes(_json_bytes(family))
    gate["gate_family_artifact_hash"] = _json_sha256(family)
    (contract.root / "validation_report.json").write_text(json.dumps(report) + "\n")

    result = check_run(contract)

    assert not result.passed
    assert any("validation_selection_split_hash: must differ" in item for item in result.invalid)


def test_subspace_contract_rejects_validation_selection_artifact_mismatch(tmp_path: Path):
    config = GpuSuiteConfig(
        output_root=tmp_path / "runs",
        systems_output_root=tmp_path / "systems",
        method="subspace",
        populations=(16,),
    )
    contract = next(item for item in gpu_suite_contracts(config) if item.name == "search_p16_subspace_r128")
    write_subspace_contract_outputs(contract)
    validation_artifact = {
        "schema_version": "validation_selection_artifact_v1",
        "selection_split_hash": "other-validation123",
        "selection_rule_hash": "other-select123",
    }
    validation_path = contract.root / "gate" / "validation_selection.json"
    validation_path.write_bytes(_json_bytes(validation_artifact))
    report = json.loads((contract.root / "validation_report.json").read_text())
    gate = report["scientific_gate_contract"]
    gate["multiple_comparison_correction"] = "separate_validation_split"
    gate["validation_selection_split_hash"] = "validation123"
    gate["validation_selection_artifact_path"] = "gate/validation_selection.json"
    gate["validation_selection_artifact_hash"] = _json_sha256(validation_artifact)
    family = json.loads((contract.root / gate["gate_family_artifact_path"]).read_text())
    family["multiple_comparison_correction"] = "separate_validation_split"
    (contract.root / gate["gate_family_artifact_path"]).write_bytes(_json_bytes(family))
    gate["gate_family_artifact_hash"] = _json_sha256(family)
    (contract.root / "validation_report.json").write_text(json.dumps(report) + "\n")

    result = check_run(contract)

    assert not result.passed
    assert any("validation_selection_artifact.selection_split_hash" in item for item in result.invalid)
    assert any("validation_selection_artifact.selection_rule_hash" in item for item in result.invalid)


def test_subspace_contract_rejects_unbacked_gate_family_observed_config(tmp_path: Path):
    config = GpuSuiteConfig(
        output_root=tmp_path / "runs",
        systems_output_root=tmp_path / "systems",
        method="subspace",
        populations=(16,),
    )
    contract = next(item for item in gpu_suite_contracts(config) if item.name == "search_p16_subspace_r128")
    write_subspace_contract_outputs(contract)
    report = json.loads((contract.root / "validation_report.json").read_text())
    gate = report["scientific_gate_contract"]
    gate["multiple_comparison_correction"] = "holm_bonferroni"
    gate["basis_rank_grid"] = [128, 512]
    family = json.loads((contract.root / gate["gate_family_artifact_path"]).read_text())
    family["multiple_comparison_correction"] = "holm_bonferroni"
    family["basis_rank_grid"] = [128, 512]
    family["observed_configs"][0]["basis_rank"] = 512
    (contract.root / gate["gate_family_artifact_path"]).write_bytes(_json_bytes(family))
    gate["gate_family_artifact_hash"] = _json_sha256(family)
    (contract.root / "validation_report.json").write_text(json.dumps(report) + "\n")

    result = check_run(contract)

    assert not result.passed
    assert any("observed_configs[1].artifact.basis_rank" in item for item in result.invalid)


def test_subspace_contract_rejects_partial_corrected_multigrid_contrast_family(tmp_path: Path):
    config = GpuSuiteConfig(
        output_root=tmp_path / "runs",
        systems_output_root=tmp_path / "systems",
        method="subspace",
        populations=(16,),
    )
    contract = next(item for item in gpu_suite_contracts(config) if item.name == "search_p16_subspace_r128")
    write_subspace_contract_outputs(contract)
    report = json.loads((contract.root / "validation_report.json").read_text())
    gate = report["scientific_gate_contract"]
    gate["multiple_comparison_correction"] = "holm_bonferroni"
    gate["K_grid"] = [1, 4]
    family = json.loads((contract.root / gate["gate_family_artifact_path"]).read_text())
    family["multiple_comparison_correction"] = "holm_bonferroni"
    family["K_grid"] = [1, 4]
    for basis_kind, artifact_path in [
        ("activation-svd", "gate/config_activation_svd_k4.json"),
        ("random-orthonormal", "gate/config_random_k4.json"),
        ("shuffled-activation-svd", "gate/config_shuffled_k4.json"),
    ]:
        config_artifact = {
            "schema_version": "scientific_gate_config_v1",
            "basis_kind": basis_kind,
            "K": 4,
            "basis_rank": 128,
            "radius": 0.01,
            "target_preset": "transformer-linears",
            "scale_mode": "relative-output-rms",
            "aggregation": "majority-vote",
            "primary_metric": "top_k_holdout_exact",
            "selection_rule_hash": "select123",
        }
        (contract.root / artifact_path).write_bytes(_json_bytes(config_artifact))
        family["observed_configs"].append(
            {
                "basis_kind": basis_kind,
                "K": 4,
                "basis_rank": 128,
                "radius": 0.01,
                "target_preset": "transformer-linears",
                "scale_mode": "relative-output-rms",
                "aggregation": "majority-vote",
                "artifact_path": artifact_path,
                "artifact_hash": _json_sha256(config_artifact),
            }
        )
    (contract.root / gate["gate_family_artifact_path"]).write_bytes(_json_bytes(family))
    gate["gate_family_artifact_hash"] = _json_sha256(family)
    (contract.root / "validation_report.json").write_text(json.dumps(report) + "\n")

    result = check_run(contract)

    assert not result.passed
    assert any("missing corrected-family contrasts" in item for item in result.invalid)


def test_subspace_contract_rejects_mixed_config_top_k_candidates(tmp_path: Path):
    config = GpuSuiteConfig(
        output_root=tmp_path / "runs",
        systems_output_root=tmp_path / "systems",
        method="subspace",
        populations=(16,),
    )
    contract = next(item for item in gpu_suite_contracts(config) if item.name == "search_p16_subspace_r128")
    write_subspace_contract_outputs(contract)
    candidates = [json.loads(line) for line in (contract.root / "candidates.jsonl").read_text().splitlines()]
    candidates[1]["basis_rank"] = 512
    candidates[1]["target_preset"] = "mlp"
    candidates[1]["rho_or_sigma_w"] = 0.5
    (contract.root / "candidates.jsonl").write_text("".join(json.dumps(row) + "\n" for row in candidates))
    top_k = json.loads((contract.root / "top_k_ensemble.json").read_text())
    top_k["K"] = 2
    top_k["candidates"] = [top_k["candidates"][0], candidates[1]]
    (contract.root / "top_k_ensemble.json").write_text(json.dumps(top_k) + "\n")
    report = json.loads((contract.root / "validation_report.json").read_text())
    gate = report["scientific_gate_contract"]
    gate["locked_K"] = 2
    gate["K_grid"] = [2]
    family = json.loads((contract.root / gate["gate_family_artifact_path"]).read_text())
    family["K_grid"] = [2]
    for config_row in family["observed_configs"]:
        config_row["K"] = 2
    (contract.root / gate["gate_family_artifact_path"]).write_bytes(_json_bytes(family))
    gate["gate_family_artifact_hash"] = _json_sha256(family)
    (contract.root / "validation_report.json").write_text(json.dumps(report) + "\n")

    result = check_run(contract)

    assert not result.passed
    assert any("locked_basis_rank: does not match top_k candidate[2]" in item for item in result.invalid)
    assert any("locked_target_preset: does not match top_k candidate[2]" in item for item in result.invalid)
    assert any("locked_radius: does not match top_k candidate[2]" in item for item in result.invalid)


def test_subspace_contract_rejects_holdout_scores_without_selector_provenance(tmp_path: Path):
    config = GpuSuiteConfig(
        output_root=tmp_path / "runs",
        systems_output_root=tmp_path / "systems",
        method="subspace",
        populations=(16,),
    )
    contract = next(item for item in gpu_suite_contracts(config) if item.name == "search_p16_subspace_r128")
    write_subspace_contract_outputs(contract)
    rows = [json.loads(line) for line in (contract.root / "candidate_scores.jsonl").read_text().splitlines()]
    rows[0]["split"] = "holdout"
    rows[0]["selection_stage"] = "selected_holdout"
    rows[0]["promoted_by_candidate_id"] = ""
    (contract.root / "candidate_scores.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    result = check_run(contract)

    assert not result.passed
    assert any("promoted_by_candidate_id" in item for item in result.invalid)


def test_subspace_contract_rejects_weak_production_scientific_gate(tmp_path: Path):
    config = GpuSuiteConfig(
        output_root=tmp_path / "runs",
        systems_output_root=tmp_path / "systems",
        method="subspace",
        populations=(16,),
    )
    contract = next(item for item in gpu_suite_contracts(config) if item.name == "search_p16_subspace_r128")
    write_subspace_contract_outputs(contract)
    report = json.loads((contract.root / "validation_report.json").read_text())
    report["scientific_gate_contract"]["confidence_interval"]["lower"] = 0.0
    (contract.root / "validation_report.json").write_text(json.dumps(report) + "\n")

    result = check_run(contract)

    assert not result.passed
    assert any("production gate requires lower > 0" in item for item in result.invalid)


def test_subspace_contract_rejects_nonpassing_validation_report(tmp_path: Path):
    config = GpuSuiteConfig(
        output_root=tmp_path / "runs",
        systems_output_root=tmp_path / "systems",
        method="subspace",
        populations=(16,),
    )
    contract = next(item for item in gpu_suite_contracts(config) if item.name == "search_p16_subspace_r128")
    write_subspace_contract_outputs(contract)
    report = json.loads((contract.root / "validation_report.json").read_text())
    report["math_tests"] = {"status": "fail", "evidence_paths": [], "failures": ["covariance mismatch"]}
    report["rng_replay_tests"] = {"status": "pass", "evidence_paths": ["missing.json"], "failures": []}
    (contract.root / "validation_report.json").write_text(json.dumps(report) + "\n")

    result = check_run(contract)

    assert not result.passed
    assert any("validation_report.json.math_tests.status" in item for item in result.invalid)
    assert any("validation_report.json.math_tests.evidence_paths" in item for item in result.invalid)
    assert any("validation_report.json.math_tests.failures" in item for item in result.invalid)
    assert any("validation_report.json.rng_replay_tests.evidence_paths" in item for item in result.invalid)


def test_subspace_contract_rejects_candidate_join_inconsistency(tmp_path: Path):
    config = GpuSuiteConfig(
        output_root=tmp_path / "runs",
        systems_output_root=tmp_path / "systems",
        method="subspace",
        populations=(16,),
    )
    contract = next(item for item in gpu_suite_contracts(config) if item.name == "search_p16_subspace_r128")
    write_subspace_contract_outputs(contract)
    rows = [json.loads(line) for line in (contract.root / "candidates.jsonl").read_text().splitlines()]
    rows[1]["candidate_id"] = rows[0]["candidate_id"]
    rows[1]["direction_seed"] = 999
    rows[1]["sign"] = "-"
    rows[1]["shard_id"] = "other"
    (contract.root / "candidates.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    score_rows = [json.loads(line) for line in (contract.root / "candidate_scores.jsonl").read_text().splitlines()]
    score_rows[0]["candidate_id"] = "invented:+:rho0.01"
    score_rows[0]["scorer_name"] = "other_scorer"
    (contract.root / "candidate_scores.jsonl").write_text("".join(json.dumps(row) + "\n" for row in score_rows))
    top_k = json.loads((contract.root / "top_k_ensemble.json").read_text())
    top_k["candidates"][0]["candidate_id"] = "invented:+:rho0.01"
    top_k["candidates"][0]["direction_seed"] = 888
    (contract.root / "top_k_ensemble.json").write_text(json.dumps(top_k) + "\n")

    result = check_run(contract)

    assert not result.passed
    assert any("duplicate candidate_id" in item for item in result.invalid)
    assert any("candidate_scores.jsonl.candidate_id" in item for item in result.invalid)
    assert any("scorer_name" in item for item in result.invalid)
    assert any("top_k_ensemble.json.candidates[1].candidate_id" in item for item in result.invalid)


def test_subspace_contract_rejects_exact_duplicate_candidates_and_scores(tmp_path: Path):
    config = GpuSuiteConfig(
        output_root=tmp_path / "runs",
        systems_output_root=tmp_path / "systems",
        method="subspace",
        populations=(16,),
    )
    contract = next(item for item in gpu_suite_contracts(config) if item.name == "search_p16_subspace_r128")
    write_subspace_contract_outputs(contract)
    candidate_lines = (contract.root / "candidates.jsonl").read_text().splitlines()
    (contract.root / "candidates.jsonl").write_text("\n".join([*candidate_lines, candidate_lines[0]]) + "\n")
    score_lines = (contract.root / "candidate_scores.jsonl").read_text().splitlines()
    (contract.root / "candidate_scores.jsonl").write_text("\n".join([*score_lines, score_lines[0]]) + "\n")

    result = check_run(contract)

    assert not result.passed
    assert any("duplicate candidate_id" in item for item in result.invalid)
    assert any("duplicate score row" in item for item in result.invalid)


def test_subspace_contract_rejects_empty_candidate_ids(tmp_path: Path):
    config = GpuSuiteConfig(
        output_root=tmp_path / "runs",
        systems_output_root=tmp_path / "systems",
        method="subspace",
        populations=(16,),
    )
    contract = next(item for item in gpu_suite_contracts(config) if item.name == "search_p16_subspace_r128")
    write_subspace_contract_outputs(contract)
    rows = [json.loads(line) for line in (contract.root / "candidates.jsonl").read_text().splitlines()]
    rows[0]["candidate_id"] = ""
    (contract.root / "candidates.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    score_rows = [json.loads(line) for line in (contract.root / "candidate_scores.jsonl").read_text().splitlines()]
    score_rows[0]["candidate_id"] = ""
    (contract.root / "candidate_scores.jsonl").write_text("".join(json.dumps(row) + "\n" for row in score_rows))
    top_k = json.loads((contract.root / "top_k_ensemble.json").read_text())
    top_k["candidates"][0]["candidate_id"] = ""
    (contract.root / "top_k_ensemble.json").write_text(json.dumps(top_k) + "\n")

    result = check_run(contract)

    assert not result.passed
    assert any("candidates.jsonl: row 1.candidate_id: empty" in item for item in result.invalid)
    assert any("candidate_scores.jsonl: row 1.candidate_id: empty" in item for item in result.invalid)
    assert any("top_k_ensemble.json.candidates[1].candidate_id: empty" in item for item in result.invalid)


def test_subspace_contract_rejects_bad_systems_report_types(tmp_path: Path):
    config = GpuSuiteConfig(
        output_root=tmp_path / "runs",
        systems_output_root=tmp_path / "systems",
        method="subspace",
        populations=(16,),
    )
    contract = next(item for item in gpu_suite_contracts(config) if item.name == "search_p16_subspace_r128")
    write_subspace_contract_outputs(contract)
    systems = json.loads((contract.root / "systems_report.json").read_text())
    systems["gpu_count"] = "1"
    systems["candidates_per_sec"] = "1.0"
    systems["top_k_ensemble_cost_multiplier"] = "huge"
    (contract.root / "systems_report.json").write_text(json.dumps(systems) + "\n")

    result = check_run(contract)

    assert not result.passed
    assert any("systems_report.json.gpu_count" in item for item in result.invalid)
    assert any("systems_report.json.candidates_per_sec" in item for item in result.invalid)
    assert any("systems_report.json.top_k_ensemble_cost_multiplier" in item for item in result.invalid)


def test_subspace_contract_rejects_invalid_basis_metadata_enums(tmp_path: Path):
    config = GpuSuiteConfig(
        output_root=tmp_path / "runs",
        systems_output_root=tmp_path / "systems",
        method="subspace",
        populations=(16,),
    )
    contract = next(item for item in gpu_suite_contracts(config) if item.name == "search_p16_subspace_r128")
    write_subspace_contract_outputs(contract)
    summary = json.loads((contract.root / "subspace_state_summary.json").read_text())
    summary["basis_kind"] = "task-biased"
    summary["basis_centering"] = "median"
    summary["basis_token_source"] = "train"
    summary["basis_split"] = "holdout_labeled"
    (contract.root / "subspace_state_summary.json").write_text(json.dumps(summary) + "\n")

    result = check_run(contract)

    assert not result.passed
    assert any("subspace_state_summary.json.basis_kind" in item for item in result.invalid)
    assert any("subspace_state_summary.json.basis_centering" in item for item in result.invalid)
    assert any("subspace_state_summary.json.basis_token_source" in item for item in result.invalid)
    assert any("subspace_state_summary.json.basis_split" in item for item in result.invalid)


def test_lora_artifacts_cannot_pass_as_subspace_run(tmp_path: Path):
    config = GpuSuiteConfig(
        output_root=tmp_path / "runs",
        systems_output_root=tmp_path / "systems",
        method="subspace",
        populations=(16,),
    )
    contract = next(item for item in gpu_suite_contracts(config) if item.name == "search_p16_subspace_r128")
    contract.root.mkdir(parents=True)
    (contract.root / "summary.json").write_text(json.dumps(_summary_for_kind("search", population=16)) + "\n")
    (contract.root / "candidate_summary.jsonl").write_text(json.dumps({"candidate": CANDIDATE, "exact_mean": 0.2}) + "\n")
    (contract.root / "per_prompt.jsonl").write_text(json.dumps({"mode": "screen", "candidate": CANDIDATE}) + "\n")
    (contract.root / "holdout_per_prompt.jsonl").write_text(json.dumps({"mode": "holdout", "candidate": CANDIDATE}) + "\n")

    result = check_run(contract)

    assert not result.passed
    assert "subspace_state.pt" in result.missing
    assert any("summary.kind" in item for item in result.invalid)


def test_run_contract_rejects_placeholder_pngs(tmp_path: Path):
    config = GpuSuiteConfig(output_root=tmp_path / "runs", systems_output_root=tmp_path / "systems", run_halving=False)
    contract = next(item for item in gpu_suite_contracts(config) if item.name == "systems_report")
    write_contract_outputs(contract)
    (contract.root / "adapter_throughput.png").write_text("placeholder\n")

    final = check_run(contract)

    assert not final.passed
    assert any("invalid PNG signature" in item for item in final.invalid)


def test_run_contract_checks_every_jsonl_row(tmp_path: Path):
    config = GpuSuiteConfig(output_root=tmp_path / "runs", systems_output_root=tmp_path / "systems", populations=(1024,))
    contract = next(item for item in gpu_suite_contracts(config) if item.name == "search_p1024_chunk32")
    write_contract_outputs(contract)
    (contract.root / "candidate_summary.jsonl").write_text(
        json.dumps({"candidate": CANDIDATE, "exact_mean": 0.2}) + "\n{}\n"
    )

    broken_rows = check_run(contract)
    assert not broken_rows.passed
    assert any("row 2 missing candidate" in item for item in broken_rows.invalid)


def test_failure_summary_is_not_a_completion_marker(tmp_path: Path):
    config = GpuSuiteConfig(output_root=tmp_path / "runs", systems_output_root=tmp_path / "systems", populations=(16,))
    spec = next(item for item in gpu_suite_specs(config) if item.name == "search_p16_chunk32")
    spec.output_path.mkdir(parents=True)
    (spec.output_path / "summary.json").write_text('{"kind": "vllm_lora_search_failure"}\n')

    assert not spec_is_complete(spec)


def test_execute_specs_dry_run_and_skip_existing(tmp_path: Path):
    config = GpuSuiteConfig(
        output_root=tmp_path / "runs",
        systems_output_root=tmp_path / "systems",
        populations=(16,),
        bench_adapters=(4,),
        run_halving=False,
    )
    specs = gpu_suite_specs(config)
    first = specs[0]
    write_completed_spec(first)

    rows = execute_specs(specs, dry_run=True)
    by_name = {row["name"]: row for row in rows}

    assert by_name[first.name]["status"] == "skipped"
    assert by_name["search_p16_chunk32"]["status"] == "dry_run"
    assert by_name["systems_report"]["status"] == "dry_run"


def test_parse_int_tuple_accepts_comma_and_space_forms():
    assert parse_int_tuple("1024,4096") == (1024, 4096)
    assert parse_int_tuple("8 16 32") == (8, 16, 32)


def test_run_suite_dry_run_cli_writes_execution_log(tmp_path: Path):
    log = tmp_path / "execution.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "optimus.cli",
            "run-suite",
            "--dry-run",
            "--no-ensure-data",
            "--root",
            str(tmp_path / "runs"),
            "--systems-out",
            str(tmp_path / "systems"),
            "--populations",
            "16",
            "--bench-adapters",
            "4",
            "--execution-log",
            str(log),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert '"dry_run": true' in result.stdout
    assert log.exists()
    assert "search_p16_chunk32" in log.read_text()


def test_validate_run_cli_respects_plan_shape(tmp_path: Path):
    root = tmp_path / "runs"
    systems = tmp_path / "systems"
    config = GpuSuiteConfig(
        output_root=root,
        systems_output_root=systems,
        populations=(1024,),
        bench_adapters=(8,),
        run_halving=False,
    )
    for contract in gpu_suite_contracts(config):
        write_contract_outputs(contract)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "optimus.cli",
            "validate-run",
            "--root",
            str(root),
            "--systems-out",
            str(systems),
            "--populations",
            "1024",
            "--bench-adapters",
            "8",
            "--strict",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert '"pass": true' in result.stdout
