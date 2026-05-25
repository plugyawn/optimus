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
                            "candidates_per_sec": 1.0,
                            "prompts_per_sec": 2.0,
                            "output_tokens_per_sec": 3.0,
                            "base_model_time_s": 1.0,
                            "qx_time_s": 0.1,
                            "lazy_delta_time_s": 0.2,
                            "scoring_time_s": 0.3,
                            "setup_time_s": 0.4,
                            "lazy_overhead_pct": 10.0,
                            "gpu_memory_allocated_bytes": 1024,
                            "candidate_batch_size": 4,
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
        "budget_policy": "per-block-equal",
        "rng_version": "gaussian_hash_v1",
        "runtime_dtype": "bf16",
    }
    summary = {
        "kind": "subspace_vllm_search",
        "backend": "vllm",
        "method": "subspace",
        "population": 16,
        "basis_hash": "basis123",
        "target_set_hash": "target123",
        "scale_mode": "relative-output-rms",
        "budget_policy": "per-block-equal",
        "rng_version": "gaussian_hash_v1",
        "candidate_routing": "row_candidate_id",
        "prefix_cache_policy": "disabled-for-search",
        "scorer_version": "countdown_v1",
        "prompt_ids_hash": "prompts123",
        "decode_config_hash": "decode123",
        "candidates_per_sec": 1.0,
        "prompts_per_sec": 2.0,
        "output_tokens_per_sec": 3.0,
        "lazy_overhead_pct": 10.0,
    }
    json_files = {
        "summary.json": summary,
        "subspace_state_summary.json": {
            "schema_version": "subspace_state_v1",
            "basis_hash": "basis123",
            "activation_sites": [{"site_id": "layer_0.attn_in"}],
            "targets": [{"target_id": "layer_0.self_attn.q_proj"}],
        },
        "top_k_ensemble.json": {
            "ensemble_kind": "lazy_top_k",
            "schema_version": "top_k_ensemble_v1",
            "aggregation": "majority-vote",
            "tie_break_policy": "lowest_candidate_id",
            "selection_rule": "screen_top_k_fixed_config",
            "K": 1,
            "candidates": [candidate],
            "basis_hash": "basis123",
            "target_set_hash": "target123",
            "scorer_version": "countdown_v1",
            "prompt_ids_hash": "prompts123",
            "runtime_config_hash": "runtime123",
            "decode_config_hash": "decode123",
        },
        "validation_report.json": {
            "schema_version": "subspace_validation_report_v1",
            "scientific_gate": {},
            "drift_diagnostics": {},
            "diversity_metrics": {},
        },
        "systems_report.json": {
            "schema_version": "subspace_systems_report_v1",
            "candidates_per_sec": 1.0,
            "prompts_per_sec": 2.0,
            "output_tokens_per_sec": 3.0,
            "base_model_time_s": 1.0,
            "qx_time_s": 0.1,
            "lazy_delta_time_s": 0.2,
            "scoring_time_s": 0.3,
            "setup_time_s": 0.4,
            "lazy_overhead_pct": 10.0,
            "gpu_memory_allocated_bytes": 1024,
            "candidate_batch_size": 4,
        },
    }
    for rel in contract.required_files:
        path = contract.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if rel == "subspace_state.pt":
            path.write_bytes(b"subspace-state")
        elif rel == "candidates.jsonl":
            path.write_text("".join(json.dumps(candidate | {"candidate_id": f"seed{idx}:+:rho0.01", "direction_seed": idx}) + "\n" for idx in range(1, 17)))
        elif rel == "candidate_scores.jsonl":
            path.write_text("".join(json.dumps({"candidate_id": f"seed{idx}:+:rho0.01", "screen_score": 0.1, "scorer_version": "countdown_v1", "prompt_ids_hash": "prompts123"}) + "\n" for idx in range(1, 17)))
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
    assert "--rho-grid" in search["command"]
    assert "--rank" not in search["command"]
    assert "--sigma" not in search["command"]
    assert "--max-loras" not in search["command"]
    assert not any("vllm-search" in command or "vllm-bench" in command or "vllm-halving" in command for command in commands)


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
            "--skip-halving",
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
            "--skip-halving",
            "--strict",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert '"pass": true' in result.stdout
