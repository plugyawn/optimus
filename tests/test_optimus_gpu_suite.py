from __future__ import annotations

from pathlib import Path

import subprocess
import sys

from optimus.evaluation.validation import check_run, gpu_suite_contracts, summary_payload
from optimus.runs.gpu_suite import GpuSuiteConfig, execute_specs, gpu_suite_specs, parse_int_tuple, plan_payload, spec_is_complete


def test_gpu_suite_specs_include_p1024_and_p4096_searches(tmp_path: Path):
    config = GpuSuiteConfig(output_root=tmp_path / "runs", systems_output_root=tmp_path / "systems")

    specs = gpu_suite_specs(config)
    names = {spec.name for spec in specs}

    assert "search_p1024_chunk8" in names
    assert "search_p4096_chunk8" in names
    assert "halving_p1024_stage8_surv64" in names
    assert "systems_report" in names


def test_plan_payload_serializes_commands(tmp_path: Path):
    config = GpuSuiteConfig(output_root=tmp_path / "runs", systems_output_root=tmp_path / "systems")

    payload = plan_payload(config)
    search = next(run for run in payload["runs"] if run["name"] == "search_p4096_chunk8")

    assert search["kind"] == "search"
    assert search["command"][:2] == ["optimus", "vllm-search"]
    assert "--population" in search["command"]
    assert "4096" in search["command"]
    assert "--tensor-parallel-size" in search["command"]
    assert "8" in search["command"]


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


def test_run_contract_checks_missing_and_present_files(tmp_path: Path):
    config = GpuSuiteConfig(output_root=tmp_path / "runs", systems_output_root=tmp_path / "systems")
    contract = next(item for item in gpu_suite_contracts(config) if item.name == "search_p1024_chunk8")

    initial = check_run(contract)
    assert not initial.passed
    assert "summary.json" in initial.missing

    contract.root.mkdir(parents=True)
    for rel in contract.required_files:
        (contract.root / rel).write_text("{}\n")

    final = check_run(contract)
    assert final.passed
    assert summary_payload([final])["pass"] is True


def test_failure_summary_is_not_a_completion_marker(tmp_path: Path):
    config = GpuSuiteConfig(output_root=tmp_path / "runs", systems_output_root=tmp_path / "systems", populations=(16,))
    spec = next(item for item in gpu_suite_specs(config) if item.name == "search_p16_chunk8")
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
    first.output_path.mkdir(parents=True)
    (first.output_path / "summary.json").write_text("{}\n")

    rows = execute_specs(specs, dry_run=True)
    by_name = {row["name"]: row for row in rows}

    assert by_name[first.name]["status"] == "skipped"
    assert by_name["search_p16_chunk8"]["status"] == "dry_run"
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
    assert "search_p16_chunk8" in log.read_text()


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
        contract.root.mkdir(parents=True, exist_ok=True)
        for rel in contract.required_files:
            (contract.root / rel).write_text("{}\n")

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
