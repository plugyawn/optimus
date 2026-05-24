from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from optimus.evaluation.release import FORBIDDEN_PACKAGE, FORBIDDEN_REPO, build_release_checks, summary


CANDIDATE = "lora:isotropic:seed1:s0.0075:sign1:r8:tq_proj,v_proj"
PNG_1X1 = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02\xfeA\xba\xa3\x8b\x00\x00\x00\x00IEND\xaeB`\x82"


def valid_bench_summary(adapters: int) -> dict:
    return {
        "kind": "vllm_lora_bench",
        "method": "lora",
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "family": "isotropic",
        "adapters": adapters,
        "prompts": 64,
        "rank": 8,
        "sigma": 0.0075,
        "seed": 2468,
        "targets": ["q_proj", "v_proj"],
        "max_new_tokens": 32,
        "tensor_parallel_size": 8,
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
        "model": "Qwen/Qwen2.5-3B-Instruct",
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
        "tensor_parallel_size": 8,
        "chunk_adapters": 8,
        "max_loras": 8,
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
    docs = root / "docs"
    docs.mkdir()
    for name in ["api.md", "gpu_suite.md", "index.md", "optimus_design.md", "release_checklist.md"]:
        (docs / name).write_text(f"# {name}\n\nUse `optimus` commands.\n")
    (root / "README.md").write_text("# Optimus\n\nUse `optimus` commands.\n")
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
        run = gpu / f"search_p{population}_chunk8"
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
    (systems / "full_search.csv").write_text("suite,run,population,candidate_sec\noptimus_gpu_suite,search_p1024_chunk8,1024,1\n")
    (systems / "best_of_n.csv").write_text("suite,run,n,best_screen_exact\noptimus_gpu_suite,search_p1024_chunk8,1,0.2\n")
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
        remote="https://github.com/plugyawn/optimus.git",
    )
    payload = summary(checks)
    failed = {check["name"] for check in payload["checks"] if not check["passed"]}

    assert payload["pass"] is False
    assert "repo_has_no_top_level_old_namespace" in failed
    assert "repo_has_no_archive_experiment_tree" in failed


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
            "--skip-halving",
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
