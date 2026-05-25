from __future__ import annotations

import json
from pathlib import Path

import subprocess
import sys

from optimus.evaluation.validation import check_run, gpu_suite_contracts, summary_payload
from optimus.runs.gpu_suite import GpuSuiteConfig, execute_specs, gpu_suite_specs, parse_int_tuple, plan_payload, spec_is_complete


CANDIDATE = "lora:isotropic:seed1:s0.0075:sign1:r8:tq_proj,v_proj"
PNG_1X1 = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02\xfeA\xba\xa3\x8b\x00\x00\x00\x00IEND\xaeB`\x82"


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
        }
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
                            "candidate_batch_size": 4,
                            "candidate_shard_id": "single",
                            "gpu_model": "test-gpu",
                            "gpu_count": 1,
                            "gpu_memory_allocated_bytes": 1024,
                            "gpu_memory_reserved_bytes": 2048,
                            "base_model_time_s": 1.0,
                            "qx_time_s": 0.1,
                            "lazy_delta_time_s": 0.2,
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
        "population": 16,
        "basis_hash": "basis123",
        "target_set_hash": "target123",
        "basis_collection_config_hash": "basisconfig123",
        "subspace_state_hash": "statehash123",
        "candidate_scores_hash": "scorehash123",
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
    validation_report = {
        "schema_version": "validation_report_v1",
        **artifact_provenance,
        **{
            key: {"status": "pass", "evidence_paths": ["summary.json"], "failures": []}
            for key in [
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
            "layers": "all",
            "basis_kind": "activation-svd",
            "basis_centering": "none",
            "basis_token_source": "prefill",
            "basis_split": "train",
            "activation_sites": [
                {
                    "site_id": "layer_0.attn_in",
                    "input_dim": 16,
                    "requested_rank": 8,
                    "effective_rank": 8,
                    "basis_tensor_key": "basis/layer_0.attn_in",
                    "singular_values": [1.0, 0.5],
                    "captured_energy": 0.9,
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
            "subspace_state_hash": "statehash123",
            "scale_mode": "relative-output-rms",
            "rho_or_sigma_w": 0.01,
            "budget_policy": "per-block-equal",
            "target_set_hash": "target123",
            "candidate_scores_hash": "scorehash123",
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
            "candidate_batch_size": 4,
            "candidate_shard_id": "single",
            "gpu_model": "test-gpu",
            "gpu_count": 1,
            "gpu_memory_allocated_bytes": 1024,
            "gpu_memory_reserved_bytes": 2048,
            "base_model_time_s": 1.0,
            "qx_time_s": 0.1,
            "lazy_delta_time_s": 0.2,
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
        },
    }
    json_files["validation_report.json"]["scientific_gate_contract"].update(
        {
            "locked_config_hash": "locked123",
            "selection_rule_hash": "select123",
            "primary_metric": "top_k_holdout_exact",
            "multiple_comparison_correction": "none_predeclared_single_config",
            "basis_kind": "activation-svd",
            "control_basis_kinds": ["random-orthonormal", "shuffled-activation-svd"],
            "comparison": "activation_svd_minus_best_control",
            "gate_type": "non-inferiority",
            "epsilon": 0.0,
            "confidence_interval": {"lower": 0.0, "upper": 0.1},
        }
    )
    for rel in contract.required_files:
        path = contract.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if rel == "subspace_state.pt":
            path.write_bytes(b"subspace-state")
        elif rel == "candidates.jsonl":
            path.write_text("".join(json.dumps(candidate | {"candidate_id": f"seed{idx}:+:rho0.01", "direction_seed": idx}) + "\n" for idx in range(1, 17)))
        elif rel == "candidate_scores.jsonl":
            path.write_text(
                "".join(
                    json.dumps(
                        {
                            "candidate_id": f"seed{idx}:+:rho0.01",
                            "split": "screen",
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
            )
        elif rel in json_files:
            path.write_text(json.dumps(json_files[rel]) + "\n")
        else:
            path.write_text("placeholder\n")


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
