from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from optimus.evaluation.release import FORBIDDEN_PACKAGE, FORBIDDEN_REPO, build_release_checks, summary, systems_report_checks


CANDIDATE = "lora:isotropic:seed1:s0.0075:sign1:r8:tq_proj,v_proj"
PNG_1X1 = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02\xfeA\xba\xa3\x8b\x00\x00\x00\x00IEND\xaeB`\x82"


def valid_bench_summary(adapters: int) -> dict:
    return {
        "kind": "vllm_lora_bench",
        "method": "lora",
        "model": "Qwen/Qwen3-4B",
        "family": "isotropic",
        "adapters": adapters,
        "prompts": 64,
        "rank": 8,
        "sigma": 0.0075,
        "seed": 2468,
        "targets": ["q_proj", "v_proj"],
        "max_new_tokens": 32,
        "tensor_parallel_size": 1,
        "adapter_build_s": 1.0,
        "load_s": 1.0,
        "lora_tokens_per_sec": None,
        "mixed_tokens_per_sec": 10.0,
        "mixed_prompts_per_sec": 2.0,
    }


def valid_search_summary(population: int) -> dict:
    return {
        "kind": "vllm_lora_search",
        "method": "lora",
        "model": "Qwen/Qwen3-4B",
        "family": "isotropic",
        "population": population,
        "rank": 8,
        "sigma": 0.0075,
        "seed": 2468,
        "targets": ["q_proj", "v_proj"],
        "screen_prompts": 64,
        "holdout_prompts": 256,
        "promote": 64,
        "max_new_tokens": 32,
        "tensor_parallel_size": 1,
        "chunk_adapters": 32,
        "max_loras": 32,
        "max_cpu_loras": 8192,
        "antithetic": True,
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


def write_minimal_release_tree(root: Path, *, include_old_package: bool = False) -> tuple[Path, Path]:
    package_include = f'["optimus*", "{FORBIDDEN_PACKAGE}*"]' if include_old_package else '["optimus*"]'
    (root / "pyproject.toml").write_text(
        f"""
[project]
name = "optimus"
version = "0.1.0"

[project.scripts]
optimus = "optimus.cli:main"

[tool.setuptools.packages.find]
include = {package_include}
""".lstrip()
    )
    package = root / "optimus"
    package.mkdir()
    (package / "__init__.py").write_text("__version__ = '0.1.0'\n")
    (package / "cli.py").write_text(
        'SUPPORTED_COMMANDS = {"search": "optimus.commands.search", "bench": "optimus.commands.bench"}\n'
    )
    docs = root / "docs"
    docs.mkdir()
    for name in [
        "api.md",
        "gpu_suite.md",
        "index.md",
        "optimus_design.md",
        "evaluation.md",
        "prime_gpu_runbook.md",
        "release_checklist.md",
    ]:
        (docs / name).write_text(f"# {name}\n\nUse `optimus` commands.\n")
    (docs / "full_model_lazy_kernel_design.md").write_text(
        "# Design\n\n`ActivationSite` uses `subspace_state_payload_v1` and v1 public target selection uses `--target-preset`.\n"
    )
    (docs / "subspace_implementation_roadmap.md").write_text(
        "# Roadmap\n\nPhase 1 adds `ActivationSite` and writes `subspace_state.pt` with payload schema `subspace_state_payload_v1`.\n"
    )
    (root / "README.md").write_text("# Optimus\n\nUse `optimus` commands.\n")
    subspace = package / "subspace"
    subspace.mkdir()
    (subspace / "__init__.py").write_text("class ActivationSite:\n    pass\n")
    systems = root / "results" / "report" / "optimus_systems"
    systems.mkdir(parents=True)
    (systems / "report.md").write_text("# Report\n\nScreen-selected heldout transfer is checked.\n")
    (systems / "quality_scaling.csv").write_text(
        "screen_selected_holdout_exact,screen_selected_holdout_delta_vs_base,promoted_holdout_oracle_exact,promoted_holdout_oracle_delta_vs_base\n"
        "0.1,0.01,0.2,0.11\n"
    )
    gpu = root / "results" / "optimus_gpu_suite"
    (gpu / "bench_a8_p64").mkdir(parents=True)
    (gpu / "bench_a8_p64" / "summary.json").write_text(json.dumps(valid_bench_summary(8)) + "\n")
    for name in ["adapter_rows.jsonl", "per_prompt.jsonl"]:
        (gpu / "bench_a8_p64" / name).write_text(json.dumps({"candidate": CANDIDATE, "exact_mean": 0.2, "mode": "mixed"}) + "\n")
    for population in [1024, 4096]:
        run = gpu / f"search_p{population}_chunk32"
        run.mkdir()
        (run / "summary.json").write_text(json.dumps(valid_search_summary(population)) + "\n")
        (run / "candidate_summary.jsonl").write_text(
            "".join(json.dumps({"candidate": CANDIDATE, "exact_mean": 0.2, "adapter_index": idx}) + "\n" for idx in range(population))
        )
        for name in ["per_prompt.jsonl", "holdout_per_prompt.jsonl"]:
            (run / name).write_text(json.dumps({"candidate": CANDIDATE, "exact_mean": 0.2, "mode": "screen"}) + "\n")
    for name in [
        "bench.csv",
        "adapter_throughput.png",
        "full_search.csv",
        "full_search_candidate_sec.png",
        "best_of_n.csv",
        "best_of_n.png",
        "quality_scaling.png",
        "token_throughput.png",
        "halving_tradeoff.png",
        "halving.csv",
    ]:
        if name.endswith(".png"):
            (systems / name).write_bytes(PNG_1X1)
        else:
            (systems / name).write_text("placeholder\n")
    (systems / "bench.csv").write_text("suite,run,adapters,mixed_tokens_per_sec\noptimus_gpu_suite,bench_a8_p64,8,10\n")
    (systems / "full_search.csv").write_text("suite,run,population,candidate_sec\noptimus_gpu_suite,search_p1024_chunk32,1024,1\n")
    (systems / "best_of_n.csv").write_text("suite,run,n,best_screen_exact\noptimus_gpu_suite,search_p1024_chunk32,1,0.2\n")
    (systems / "quality_scaling.csv").write_text(
        "screen_selected_holdout_exact,screen_selected_holdout_delta_vs_base,promoted_holdout_oracle_exact,promoted_holdout_oracle_delta_vs_base\n"
        "0.1,0.01,0.2,0.11\n"
    )
    (systems / "parity.csv").write_text(
        "suite,run,trusted_name,candidate_name,n_common,pass,pass_protocol,pass_base_rows,pass_adapter_tensors,pass_output_diff\n"
        "backend_parity_gate,gate,peft,vllm,1,true,true,true,true,true\n"
    )
    return gpu, systems


def test_release_check_passes_clean_optimus_tree(tmp_path: Path):
    gpu, systems = write_minimal_release_tree(tmp_path)

    checks = build_release_checks(
        root=tmp_path,
        systems_out=systems,
        gpu_root=gpu,
        populations=(1024, 4096),
        bench_adapters=(8,),
        run_halving=False,
        method="lora",
        remote="https://github.com/plugyawn/optimus.git",
    )

    payload = summary(checks)
    assert payload["pass"] is True


def test_release_check_flags_old_package_and_old_remote(tmp_path: Path):
    gpu, systems = write_minimal_release_tree(tmp_path, include_old_package=True)

    checks = build_release_checks(
        root=tmp_path,
        systems_out=systems,
        gpu_root=gpu,
        populations=(1024, 4096),
        bench_adapters=(8,),
        run_halving=False,
        method="lora",
        remote=f"https://github.com/plugyawn/{FORBIDDEN_REPO}.git",
    )
    payload = summary(checks)
    failed = {check["name"] for check in payload["checks"] if not check["passed"]}

    assert payload["pass"] is False
    assert "published_package_excludes_old_namespace" in failed
    assert "github_remote_is_optimus" in failed


def test_release_check_flags_old_root_shape(tmp_path: Path):
    gpu, systems = write_minimal_release_tree(tmp_path)
    old_namespace = tmp_path / FORBIDDEN_PACKAGE
    old_namespace.mkdir()
    (old_namespace / "__init__.py").write_text("")
    archive = tmp_path / "docs" / "archive"
    archive.mkdir()

    checks = build_release_checks(
        root=tmp_path,
        systems_out=systems,
        gpu_root=gpu,
        populations=(1024, 4096),
        bench_adapters=(8,),
        run_halving=False,
        method="lora",
        remote="https://github.com/plugyawn/optimus.git",
    )
    payload = summary(checks)
    failed = {check["name"] for check in payload["checks"] if not check["passed"]}

    assert payload["pass"] is False
    assert "repo_has_no_top_level_old_namespace" in failed
    assert "repo_has_no_archive_experiment_tree" in failed


def test_release_check_requires_all_public_docs(tmp_path: Path):
    gpu, systems = write_minimal_release_tree(tmp_path)
    (tmp_path / "docs" / "evaluation.md").unlink()

    checks = build_release_checks(
        root=tmp_path,
        systems_out=systems,
        gpu_root=gpu,
        populations=(1024, 4096),
        bench_adapters=(8,),
        run_halving=False,
        method="lora",
        remote="https://github.com/plugyawn/optimus.git",
    )
    failed = {check.name for check in checks if not check.passed}

    assert "public_docs_present" in failed


def test_release_check_scans_source_of_truth_docs_for_legacy_public_surface(tmp_path: Path):
    gpu, systems = write_minimal_release_tree(tmp_path)
    (tmp_path / "docs" / "full_model_lazy_kernel_design.md").write_text("Use optimus vllm-search as the production route.\n")
    (tmp_path / "docs" / "subspace_implementation_roadmap.md").write_text("Load family_state.pt before search.\n")

    checks = build_release_checks(
        root=tmp_path,
        systems_out=systems,
        gpu_root=gpu,
        populations=(1024, 4096),
        bench_adapters=(8,),
        run_halving=False,
        method="lora",
        remote="https://github.com/plugyawn/optimus.git",
    )
    by_name = {check.name: check for check in checks}

    assert not by_name["public_docs_do_not_promote_legacy_subspace_surface"].passed
    assert "full_model_lazy_kernel_design.md" in by_name["public_docs_do_not_promote_legacy_subspace_surface"].detail
    assert "subspace_implementation_roadmap.md" in by_name["public_docs_do_not_promote_legacy_subspace_surface"].detail


def test_release_check_locks_subspace_source_of_truth_identifiers(tmp_path: Path):
    gpu, systems = write_minimal_release_tree(tmp_path)
    (tmp_path / "docs" / "full_model_lazy_kernel_design.md").write_text(
        "Every site is an `ActivationSiteSpec`; extend through `--targets`.\n"
    )
    (tmp_path / "docs" / "subspace_implementation_roadmap.md").write_text(
        "Implement `subspace_state.pt` read/write with schema `subspace_state_v1`; use disabled/candidate-keyed cache policy.\n"
    )

    checks = build_release_checks(
        root=tmp_path,
        systems_out=systems,
        gpu_root=gpu,
        populations=(1024, 4096),
        bench_adapters=(8,),
        run_halving=False,
        method="lora",
        remote="https://github.com/plugyawn/optimus.git",
    )
    by_name = {check.name: check for check in checks}

    assert not by_name["subspace_source_of_truth_identifiers_consistent"].passed
    assert "ActivationSiteSpec" in by_name["subspace_source_of_truth_identifiers_consistent"].detail
    assert "design_doc_--targets" in by_name["subspace_source_of_truth_identifiers_consistent"].detail
    assert "subspace_state.pt/subspace_state_v1" in by_name["subspace_source_of_truth_identifiers_consistent"].detail


def test_release_check_rejects_legacy_package_level_exports(tmp_path: Path):
    gpu, systems = write_minimal_release_tree(tmp_path)
    serving = tmp_path / "optimus" / "serving"
    serving.mkdir()
    (serving / "__init__.py").write_text("__all__ = ['AdapterSpec', 'score_rows', 'make_sampling_params']\n")
    (serving / "vllm.py").write_text("def run_vllm_search(): pass\n")
    search = tmp_path / "optimus" / "search"
    search.mkdir()
    (search / "__init__.py").write_text("__all__ = ['run_peft_search']\n")
    (serving / "halving.py").write_text("def run_halving(): pass\n")

    checks = build_release_checks(
        root=tmp_path,
        systems_out=systems,
        gpu_root=gpu,
        populations=(1024, 4096),
        bench_adapters=(8,),
        run_halving=False,
        method="lora",
        remote="https://github.com/plugyawn/optimus.git",
    )
    by_name = {check.name: check for check in checks}

    assert not by_name["public_package_surface_excludes_legacy_subspace_internals"].passed
    assert "AdapterSpec" in by_name["public_package_surface_excludes_legacy_subspace_internals"].detail
    assert "run_peft_search" in by_name["public_package_surface_excludes_legacy_subspace_internals"].detail
    assert "halving.py" in by_name["public_package_surface_excludes_legacy_subspace_internals"].detail
    assert "vllm.py" in by_name["public_package_surface_excludes_legacy_subspace_internals"].detail


def _write_subspace_systems_out(root: Path, *, timing: bool = True, numeric_as_string: bool = False) -> Path:
    systems = root / "systems"
    systems.mkdir()
    source_run = root / "run"
    source_run.mkdir()
    source_report = source_run / "systems_report.json"
    source_report.write_text(json.dumps({"schema_version": "subspace_systems_report_v1"}) + "\n")
    if timing:
        (source_run / "timing_trace.jsonl").write_text(json.dumps({"elapsed_s": 0.1, "cuda_synchronized": True}) + "\n")
    payload = {
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
        "population": 128,
        "target_preset": "transformer-linears",
        "basis_rank": 128,
        "kernel": "torch",
        "candidate_batch_size": 4,
        "candidate_shard_id": "single",
        "gpu_model": "test-gpu",
        "gpu_count": "1" if numeric_as_string else 1,
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
    (systems / "report.md").write_text("# Report\n")
    (systems / "systems_report.json").write_text(json.dumps(payload) + "\n")
    (systems / "subspace_systems.csv").write_text(
        "source_run_dir,candidate_batch_size,population,target_preset,basis_rank,kernel,candidates_per_sec,prompts_per_sec,output_tokens_per_sec,lazy_overhead_pct,base_model_time_s,qx_time_s,lazy_delta_time_s,top_k_ensemble_cost_multiplier,screen_score,holdout_score,screen_to_holdout_drop,diversity_metrics,random_q_control,shuffled_q_control,antithetic_odd_even\n"
        "run,4,128,qv,128,torch,2.2,2.0,3.0,10.0,1.0,0.05,0.1,1.0,0.1,0.2,-0.1,{},{},{},{}\n"
        "run,4,128,attn-qkvo,128,torch,2.0,2.0,3.0,10.0,1.0,0.05,0.1,1.0,0.1,0.2,-0.1,{},{},{},{}\n"
        "run,4,128,mlp,128,torch,1.8,2.0,3.0,10.0,1.0,0.05,0.1,1.0,0.1,0.2,-0.1,{},{},{},{}\n"
        "run,4,128,transformer-linears,128,torch,1.2,2.0,3.0,10.0,1.0,0.05,0.1,1.0,0.1,0.2,-0.1,{},{},{},{}\n"
    )
    return systems


def test_subspace_release_check_requires_measured_systems_evidence(tmp_path: Path):
    systems = _write_subspace_systems_out(tmp_path, timing=False)

    checks = systems_report_checks(systems, method="subspace")
    by_name = {check.name: check for check in checks}

    assert not by_name["subspace_systems_report_fields_present"].passed
    assert "missing_timing_evidence" in by_name["subspace_systems_report_fields_present"].detail


def test_subspace_release_check_rejects_string_numeric_systems_fields(tmp_path: Path):
    systems = _write_subspace_systems_out(tmp_path, numeric_as_string=True)

    checks = systems_report_checks(systems, method="subspace")
    by_name = {check.name: check for check in checks}

    assert not by_name["subspace_systems_report_fields_present"].passed
    assert "gpu_count" in by_name["subspace_systems_report_fields_present"].detail


def test_subspace_release_check_rejects_invalid_source_report(tmp_path: Path):
    systems = _write_subspace_systems_out(tmp_path)
    ((tmp_path / "run") / "systems_report.json").write_text("{}\n")

    checks = systems_report_checks(systems, method="subspace")
    by_name = {check.name: check for check in checks}

    assert not by_name["subspace_systems_report_fields_present"].passed
    assert "source_report_schema" in by_name["subspace_systems_report_fields_present"].detail


def test_subspace_release_check_rejects_incomplete_p128_speed_gate(tmp_path: Path):
    systems = _write_subspace_systems_out(tmp_path)
    (systems / "subspace_systems.csv").write_text(
        "source_run_dir,candidate_batch_size,population,target_preset,basis_rank,kernel,candidates_per_sec,prompts_per_sec,output_tokens_per_sec,lazy_overhead_pct,base_model_time_s,qx_time_s,lazy_delta_time_s,top_k_ensemble_cost_multiplier,screen_score,holdout_score,screen_to_holdout_drop,diversity_metrics,random_q_control,shuffled_q_control,antithetic_odd_even\n"
        "run,4,128,transformer-linears,128,torch,1.0,2.0,3.0,10.0,1.0,0.2,0.2,1.0,0.1,0.2,-0.1,{},{},{},{}\n"
    )

    checks = systems_report_checks(systems, method="subspace")
    by_name = {check.name: check for check in checks}

    assert not by_name["subspace_p128_speed_gate_enforced"].passed
    assert "missing target presets" in by_name["subspace_p128_speed_gate_enforced"].detail
    assert "qx_plus_lazy_delta_overhead" in by_name["subspace_p128_speed_gate_enforced"].detail


def test_release_check_accepts_verified_prime_zero_ledger(tmp_path: Path):
    gpu, systems = write_minimal_release_tree(tmp_path)
    ledger = tmp_path / ".opencode" / "prime-gpu-ledger.md"
    ledger.parent.mkdir()
    ledger.write_text("- pod_id: p1\n  status: terminated; verified prime pods list total 0\n")

    checks = build_release_checks(
        root=tmp_path,
        systems_out=systems,
        gpu_root=gpu,
        populations=(1024, 4096),
        bench_adapters=(8,),
        run_halving=False,
        method="lora",
        remote="https://github.com/plugyawn/optimus.git",
    )
    by_name = {check.name: check for check in checks}

    assert by_name["prime_ledger_local_check"].passed


def test_release_check_rejects_terminated_prime_entry_without_entry_local_zero_verification(tmp_path: Path):
    gpu, systems = write_minimal_release_tree(tmp_path)
    ledger = tmp_path / ".opencode" / "prime-gpu-ledger.md"
    ledger.parent.mkdir()
    ledger.write_text(
        "- pod_id: p1\n"
        "  status: terminated\n"
        "\n"
        "- pod_id: p2\n"
        "  status: terminated; verified prime pods list total 0\n"
    )

    checks = build_release_checks(
        root=tmp_path,
        systems_out=systems,
        gpu_root=gpu,
        populations=(1024, 4096),
        bench_adapters=(8,),
        run_halving=False,
        method="lora",
        remote="https://github.com/plugyawn/optimus.git",
    )
    by_name = {check.name: check for check in checks}

    assert not by_name["prime_ledger_local_check"].passed
    assert "entry 1" in by_name["prime_ledger_local_check"].detail


def test_release_check_cli_is_lightweight(tmp_path: Path):
    gpu, systems = write_minimal_release_tree(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "optimus.cli",
            "release-check",
            "--root",
            str(tmp_path),
            "--gpu-root",
            str(gpu),
            "--systems-out",
            str(systems),
            "--populations",
            "1024,4096",
            "--bench-adapters",
            "8",
            "--method",
            "lora",
            "--remote-url",
            "https://github.com/plugyawn/optimus.git",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["pass"] is True
    assert "PyTorch" not in result.stderr
    assert "NumPy" not in result.stderr
