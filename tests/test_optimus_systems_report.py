from __future__ import annotations

import json
from pathlib import Path

from optimus.evaluation.systems import (
    best_of_n_rows,
    full_search_rows,
    halving_rows,
    main as systems_report_main,
    parity_rows,
    quality_rows,
    systems_summaries,
)
from optimus.evaluation.validation import SUBSPACE_SYSTEMS_FIELDS


def _subspace_systems_payload() -> dict:
    payload = {
        "schema_version": "subspace_systems_report_v1",
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
        "decode_config_hash": "decode123",
        "warmup_policy": "one_warmup_batch",
        "cuda_sync_policy": "sync_timed_regions",
        "benchmark_kind": "subspace",
        "population": 128,
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
    }
    assert not [field for field in SUBSPACE_SYSTEMS_FIELDS if field not in payload]
    return payload


def test_systems_report_discovers_optimus_p4096_runs(tmp_path: Path):
    run = tmp_path / "optimus_gpu_suite" / "search_p4096_chunk8"
    run.mkdir(parents=True)
    (run / "summary.json").write_text(
        json.dumps(
            {
                "kind": "vllm_lora_search",
                "population": 4096,
                "screen_prompts": 64,
                "holdout_prompts": 128,
                "chunk_adapters": 8,
                "max_loras": 8,
                "max_new_tokens": 32,
                "tensor_parallel_size": 8,
                "base_screen_exact": 0.10,
                "base_holdout_exact": 0.09,
                "top_screen": [{"candidate": "c2", "exact_mean": 0.20}],
                "top_holdout": [{"candidate": "c2", "exact_mean": 0.16}],
                "best_ensemble_holdout_exact": 0.18,
                "best_strict_ensemble_holdout_exact": 0.17,
                "candidate_sec": 3.25,
                "screen_prompts_per_sec": 208.0,
                "screen_tokens_per_sec": 4096.0,
                "holdout_tokens_per_sec": 3500.0,
                "best_tokens_per_sec": 4300.0,
                "eval_elapsed_s": 1260.0,
                "load_s": 80.0,
            }
        )
        + "\n"
    )
    (run / "candidate_summary.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"adapter_index": 0, "candidate": "c0", "exact_mean": 0.08}),
                json.dumps({"adapter_index": 1, "candidate": "c1", "exact_mean": 0.12}),
                json.dumps({"adapter_index": 2, "candidate": "c2", "exact_mean": 0.20}),
            ]
        )
        + "\n"
    )

    rows = systems_summaries(tmp_path)
    full = full_search_rows(rows)
    best = best_of_n_rows(full)

    assert len(rows) == 1
    assert full[0]["suite"] == "optimus_gpu_suite"
    assert full[0]["run"] == "search_p4096_chunk8"
    assert full[0]["population"] == 4096
    assert full[0]["tensor_parallel_size"] == 8
    assert full[0]["screen_tokens_per_sec"] == 4096.0
    assert [row["best_screen_exact"] for row in best] == [0.08, 0.12, 0.20]

    direct_rows = systems_summaries(tmp_path / "optimus_gpu_suite")
    assert len(direct_rows) == 1
    assert full_search_rows(direct_rows)[0]["run"] == "search_p4096_chunk8"


def test_systems_report_writes_best_of_n_and_scaling_outputs(tmp_path: Path):
    search = tmp_path / "optimus_gpu_suite" / "search_p1024_chunk8"
    search.mkdir(parents=True)
    (search / "summary.json").write_text(
        json.dumps(
            {
                "kind": "vllm_lora_search",
                "population": 1024,
                "screen_prompts": 64,
                "holdout_prompts": 128,
                "chunk_adapters": 8,
                "max_loras": 8,
                "max_new_tokens": 32,
                "tensor_parallel_size": 8,
                "base_screen_exact": 0.10,
                "base_holdout_exact": 0.09,
                "top_screen": [{"candidate": "c1", "exact_mean": 0.15}],
                "top_holdout": [{"candidate": "c1", "exact_mean": 0.14}],
                "best_ensemble_holdout_exact": 0.16,
                "best_strict_ensemble_holdout_exact": 0.15,
                "candidate_sec": 2.5,
                "screen_prompts_per_sec": 160.0,
                "screen_tokens_per_sec": 3000.0,
                "holdout_tokens_per_sec": 2900.0,
                "best_tokens_per_sec": 3000.0,
                "eval_elapsed_s": 1000.0,
                "load_s": 50.0,
            }
        )
        + "\n"
    )
    (search / "candidate_summary.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"adapter_index": 0, "candidate": "c0", "exact_mean": 0.11}),
                json.dumps({"adapter_index": 1, "candidate": "c1", "exact_mean": 0.15}),
            ]
        )
        + "\n"
    )
    bench = tmp_path / "optimus_gpu_suite" / "bench_a8_p64"
    bench.mkdir(parents=True)
    (bench / "summary.json").write_text(
        json.dumps(
            {
                "kind": "vllm_lora_bench",
                "adapters": 8,
                "prompts": 64,
                "tensor_parallel_size": 8,
                "max_new_tokens": 32,
                "lora_tokens_per_sec": 2500.0,
                "mixed_tokens_per_sec": 3200.0,
                "lora_prompts_per_sec": 140.0,
                "mixed_prompts_per_sec": 180.0,
                "load_s": 42.0,
            }
        )
        + "\n"
    )

    out = tmp_path / "report"
    assert systems_report_main(["--root", str(tmp_path), "--out", str(out)]) == 0

    for name in [
        "bench.csv",
        "adapter_throughput.png",
        "best_of_n.csv",
        "best_of_n.png",
        "quality_scaling.csv",
        "quality_scaling.png",
        "token_throughput.png",
    ]:
        path = out / name
        assert path.exists(), name
        assert path.stat().st_size > 0, name
    report = (out / "report.md").read_text()
    assert "parity_gates.png" not in report
    assert "screen_selected_holdout_exact" in report


def test_systems_report_writes_subspace_systems_json_from_measured_runs(tmp_path: Path):
    run = tmp_path / "optimus_gpu_suite" / "search_p128_subspace_r128"
    run.mkdir(parents=True)
    (run / "summary.json").write_text(
        json.dumps({"kind": "subspace_vllm_search", "method": "subspace", "population": 128}) + "\n"
    )
    (run / "systems_report.json").write_text(json.dumps(_subspace_systems_payload()) + "\n")
    (run / "timing_trace.jsonl").write_text(json.dumps({"event": "timed_region", "elapsed_s": 0.1, "cuda_synchronized": True}) + "\n")

    out = tmp_path / "report"
    assert systems_report_main(["--root", str(tmp_path), "--out", str(out)]) == 0

    payload = json.loads((out / "systems_report.json").read_text())
    assert payload["schema_version"] == "subspace_systems_report_v1"
    assert payload["benchmark_kind"] == "subspace"
    assert payload["prefix_cache_policy"] == "disabled-for-search"
    assert (out / payload["source_run_dir"]).resolve() == run.resolve()
    assert "source_run_dir" in (out / "subspace_systems.csv").read_text()
    assert "benchmark_kind" in (out / "subspace_systems.csv").read_text()
    assert "target_preset" in (out / "subspace_systems.csv").read_text()
    assert "Subspace Systems" in (out / "report.md").read_text()


def test_systems_report_includes_subspace_baseline_benchmark_rows(tmp_path: Path):
    suite = tmp_path / "optimus_gpu_suite"
    subspace_run = suite / "search_p128_subspace_r128"
    subspace_run.mkdir(parents=True)
    (subspace_run / "summary.json").write_text(
        json.dumps({"kind": "subspace_vllm_search", "method": "subspace", "population": 128}) + "\n"
    )
    (subspace_run / "systems_report.json").write_text(json.dumps(_subspace_systems_payload()) + "\n")
    (subspace_run / "timing_trace.jsonl").write_text(json.dumps({"event": "timed_region", "elapsed_s": 0.1, "cuda_synchronized": True}) + "\n")

    baseline_run = suite / "base_vllm_p128"
    baseline_run.mkdir()
    (baseline_run / "summary.json").write_text(json.dumps({"kind": "vllm_base_bench", "population": 128}) + "\n")
    baseline_payload = _subspace_systems_payload()
    baseline_payload["benchmark_kind"] = "base_vllm"
    baseline_payload["target_preset"] = "base"
    baseline_payload["basis_rank"] = 0
    baseline_payload["qx_time_s"] = 0.0
    baseline_payload["lazy_delta_time_s"] = 0.0
    (baseline_run / "systems_report.json").write_text(json.dumps(baseline_payload) + "\n")
    (baseline_run / "timing_trace.jsonl").write_text(json.dumps({"event": "timed_region", "elapsed_s": 0.1, "cuda_synchronized": True}) + "\n")

    out = tmp_path / "report"
    assert systems_report_main(["--root", str(tmp_path), "--out", str(out)]) == 0

    csv_text = (out / "subspace_systems.csv").read_text()
    assert "subspace" in csv_text
    assert "base_vllm" in csv_text


def test_systems_report_selects_conservative_slowest_subspace_row(tmp_path: Path):
    suite = tmp_path / "optimus_gpu_suite"
    for name, candidate_sec, target_preset in [
        ("search_p128_subspace_qv", 10.0, "qv"),
        ("search_p128_subspace_transformer_linears", 1.0, "transformer-linears"),
    ]:
        run = suite / name
        run.mkdir(parents=True)
        (run / "summary.json").write_text(
            json.dumps({"kind": "subspace_vllm_search", "method": "subspace", "population": 128, "target_preset": target_preset}) + "\n"
        )
        payload = _subspace_systems_payload()
        payload["candidates_per_sec"] = candidate_sec
        payload["target_preset"] = target_preset
        (run / "systems_report.json").write_text(json.dumps(payload) + "\n")
        (run / "timing_trace.jsonl").write_text(json.dumps({"event": "timed_region", "elapsed_s": 0.1, "cuda_synchronized": True}) + "\n")

    out = tmp_path / "report"
    assert systems_report_main(["--root", str(tmp_path), "--out", str(out)]) == 0

    selected = json.loads((out / "systems_report.json").read_text())
    assert selected["candidates_per_sec"] == 1.0
    assert selected["target_preset"] == "transformer-linears"
    assert selected["systems_selection_policy"] == "slowest_candidates_per_sec_conservative_gate"
    assert selected["all_reports_count"] == 2
    assert "qv" in (out / "subspace_systems.csv").read_text()


def test_systems_report_rejects_bad_subspace_system_types(tmp_path: Path):
    run = tmp_path / "optimus_gpu_suite" / "search_p128_subspace_r128"
    run.mkdir(parents=True)
    (run / "summary.json").write_text(
        json.dumps({"kind": "subspace_vllm_search", "method": "subspace", "population": 128}) + "\n"
    )
    payload = _subspace_systems_payload()
    payload["gpu_count"] = "1"
    payload["candidates_per_sec"] = "1.0"
    payload["top_k_ensemble_cost_multiplier"] = "huge"
    (run / "systems_report.json").write_text(json.dumps(payload) + "\n")
    (run / "timing_trace.jsonl").write_text(json.dumps({"event": "timed_region", "elapsed_s": 0.1, "cuda_synchronized": True}) + "\n")

    out = tmp_path / "report"
    assert systems_report_main(["--root", str(tmp_path), "--out", str(out)]) == 1

    report = (out / "report.md").read_text()
    assert "nonnumeric fields" in report


def test_systems_report_rejects_unsynchronized_timing_evidence(tmp_path: Path):
    run = tmp_path / "optimus_gpu_suite" / "search_p128_subspace_r128"
    run.mkdir(parents=True)
    (run / "summary.json").write_text(
        json.dumps({"kind": "subspace_vllm_search", "method": "subspace", "population": 128}) + "\n"
    )
    (run / "systems_report.json").write_text(json.dumps(_subspace_systems_payload()) + "\n")
    (run / "timing_trace.jsonl").write_text(json.dumps({"event": "timed_region", "elapsed_s": 0.1}) + "\n")

    out = tmp_path / "report"
    assert systems_report_main(["--root", str(tmp_path), "--out", str(out)]) == 1

    report = (out / "report.md").read_text()
    assert "no_cuda_synchronized_marker" in report


def test_systems_report_rejects_p128_lazy_overhead_regression(tmp_path: Path):
    run = tmp_path / "optimus_gpu_suite" / "search_p128_subspace_r128"
    run.mkdir(parents=True)
    (run / "summary.json").write_text(
        json.dumps({"kind": "subspace_vllm_search", "method": "subspace", "population": 128}) + "\n"
    )
    payload = _subspace_systems_payload()
    payload["qx_time_s"] = 0.2
    payload["lazy_delta_time_s"] = 0.2
    (run / "systems_report.json").write_text(json.dumps(payload) + "\n")
    (run / "timing_trace.jsonl").write_text(json.dumps({"event": "timed_region", "elapsed_s": 0.1, "cuda_synchronized": True}) + "\n")

    out = tmp_path / "report"
    assert systems_report_main(["--root", str(tmp_path), "--out", str(out)]) == 1

    report = (out / "report.md").read_text()
    assert "p128 qx_plus_lazy_delta_overhead" in report


def test_systems_report_fails_closed_for_subspace_without_measured_report(tmp_path: Path):
    run = tmp_path / "optimus_gpu_suite" / "search_p128_subspace_r128"
    run.mkdir(parents=True)
    (run / "summary.json").write_text(
        json.dumps({"kind": "subspace_vllm_search", "method": "subspace", "population": 128}) + "\n"
    )

    out = tmp_path / "report"
    assert systems_report_main(["--root", str(tmp_path), "--out", str(out)]) == 1

    report = (out / "report.md").read_text()
    assert "failed closed" in report
    assert "systems_report.json" in report


def test_quality_rows_separate_selected_transfer_from_holdout_oracle(tmp_path: Path):
    run = tmp_path / "optimus_gpu_suite" / "search_p4096_chunk8"
    run.mkdir(parents=True)
    (run / "summary.json").write_text(
        json.dumps(
            {
                "kind": "vllm_lora_search",
                "population": 4096,
                "screen_prompts": 64,
                "holdout_prompts": 256,
                "base_screen_exact": 0.10,
                "base_holdout_exact": 0.08,
                "top_screen": [
                    {"candidate": "screen-winner", "exact_mean": 0.21},
                    {"candidate": "holdout-winner", "exact_mean": 0.17},
                ],
                "top_holdout": [
                    {"candidate": "holdout-winner", "exact_mean": 0.16},
                    {"candidate": "screen-winner", "exact_mean": 0.07},
                ],
            }
        )
        + "\n"
    )

    quality = quality_rows(systems_summaries(tmp_path))

    assert len(quality) == 1
    row = quality[0]
    assert row["screen_selected_candidate"] == "screen-winner"
    assert row["screen_selected_exact"] == 0.21
    assert row["screen_selected_holdout_exact"] == 0.07
    assert row["screen_selected_holdout_delta_vs_base"] < 0
    assert row["promoted_holdout_oracle_candidate"] == "holdout-winner"
    assert row["promoted_holdout_oracle_exact"] == 0.16
    assert row["promoted_holdout_oracle_delta_vs_base"] == 0.08


def test_systems_report_includes_halving_and_strict_parity_rows(tmp_path: Path):
    halving = tmp_path / "optimus_gpu_suite" / "halving_p1024_stage8_surv64"
    halving.mkdir(parents=True)
    (halving / "summary.json").write_text(
        json.dumps(
            {
                "kind": "vllm_lora_halving",
                "method": "lora",
                "screen_prompts": 64,
                "stage_prompts": 8,
                "survivors": 64,
                "candidate_sec": 2.0,
                "prompt_eval_savings": 0.5,
                "top_stage": [{"candidate": "c1", "exact_mean": 0.2}],
                "top_screen": [{"candidate": "c1", "exact_mean": 0.3}],
                "top_holdout": [{"candidate": "c1", "exact_mean": 0.25}],
                "eval_elapsed_s": 100.0,
            }
        )
        + "\n"
    )
    gate = tmp_path / "backend_parity_gate" / "gate"
    gate.mkdir(parents=True)
    (gate / "summary.json").write_text(
        json.dumps(
            {
                "kind": "backend_parity_gate",
                "trusted_name": "peft",
                "candidate_name": "vllm",
                "pass": True,
                "pass_protocol": True,
                "pass_base_rows": True,
                "pass_adapter_tensors": True,
                "pass_output_diff": True,
                "ranking": {"n_common": 8, "spearman": 1.0, "top8_overlap": 8, "top8_possible": 8},
            }
        )
        + "\n"
    )

    rows = systems_summaries(tmp_path)
    staged = halving_rows(rows)
    parity = parity_rows(rows)

    assert staged[0]["run"] == "halving_p1024_stage8_surv64"
    assert staged[0]["screen_selected_holdout_exact"] == 0.25
    assert parity[0]["pass_output_diff"] is True
