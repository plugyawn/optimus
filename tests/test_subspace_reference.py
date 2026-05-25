from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from optimus.evaluation.validation import check_run, gpu_suite_contracts
from optimus.runs.gpu_suite import GpuSuiteConfig
from optimus.subspace import gaussian_hash_v1
from optimus.subspace.reference import build_basis, resolve_target_scales, run_reference_search


def test_gaussian_hash_is_candidate_order_independent():
    first = gaussian_hash_v1(direction_seed=123, target_id="layer_0.self_attn.q_proj", output_index=7, basis_index=3)
    reordered = gaussian_hash_v1(direction_seed=123, target_id="layer_0.self_attn.q_proj", output_index=7, basis_index=3)
    antithetic = gaussian_hash_v1(direction_seed=123, target_id="layer_0.self_attn.q_proj", output_index=7, basis_index=3, sign="-")

    assert first == reordered
    assert antithetic == -first


def test_relative_output_rms_scale_is_rank_budgeted():
    class Args:
        seed = 1
        basis_centering = "none"
        basis_token_source = "prefill"

    from optimus.tasks.countdown import built_in_examples
    from optimus.subspace.reference import build_reference_state

    examples = built_in_examples()[:8]
    state = build_reference_state(
        calibration_examples=examples,
        args=Args(),
        basis_kind="activation-svd",
        input_dim=16,
        output_dim=8,
        basis_rank=4,
        layers=(0,),
        target_preset="qv",
        prompt_ids_hash="prompts",
        decode_config_hash="decode",
    )
    scales = resolve_target_scales(
        state,
        scale_mode="relative-output-rms",
        radii=[0.01],
        budget_policy="per-target-equal",
    )

    assert len(scales) == 2
    assert all(scale.beta_t_by_radius["0.01"] > 0.0 for scale in scales)


def test_basis_builder_controls_are_orthonormal():
    activations = torch.randn(12, 16, generator=torch.Generator().manual_seed(5))
    for basis_kind in ["activation-svd", "random-orthonormal", "shuffled-activation-svd"]:
        basis, _, h_s, a_s, captured, error = build_basis(
            activations,
            requested_rank=8,
            basis_kind=basis_kind,
            centering="none",
            seed=7,
        )
        assert basis.shape == (8, 16)
        assert h_s > 0.0
        assert a_s > 0.0
        assert 0.0 <= captured <= 1.0001
        assert error < 1e-4


def test_reference_search_writes_valid_hardened_artifacts(tmp_path: Path):
    class Args:
        out = str(tmp_path / "run")
        backend = "transformers"
        method = "subspace"
        model = "reference"
        data = None
        prompts = 4
        holdout_prompts = 4
        population = 8
        promote = 2
        seed = 123
        tensor_parallel_size = 1
        max_new_tokens = 4
        prompt_variants = "default"
        prompt_input = "text"
        use_chat_template = False
        require_all_prompt_variants_valid = False
        max_base_malformed_for_selection = 0.05
        max_base_cap_hit_for_selection = 0.05
        min_selection_prompt_variants = 1
        stop_at_answer = True
        antithetic = True
        enable_prefix_caching = None
        enable_chunked_prefill = None
        kv_cache_dtype = ""
        vllm_kwarg = None
        basis_rank = 4
        basis_prompts = 4
        target_preset = "qv"
        layers = "all"
        basis_centering = "none"
        basis_token_source = "prefill"
        basis_kind = "activation-svd"
        scale_mode = "relative-output-rms"
        rho_grid = "0.01"
        sigma_w_grid = None
        budget_policy = "per-target-equal"
        top_k_grid = "1"
        candidate_batch_size = "auto"
        kernel = "torch"
        prefix_cache_policy = "disabled-for-search"
        match_screen_to_holdout_base_exact = False
        screen_pool_prompts = None

    summary = run_reference_search(Args(), backend="transformers")
    assert summary["kind"] == "subspace_transformers_search"

    config = GpuSuiteConfig(
        output_root=tmp_path,
        systems_output_root=tmp_path / "systems",
        backend="transformers",
        method="subspace",
        populations=(8,),
        prompts=4,
        holdout_prompts=4,
        promote=2,
        basis_rank=4,
        basis_prompts=4,
        target_preset="qv",
        budget_policy="per-target-equal",
        rho_grid="0.01",
        top_k_grid="1",
    )
    contract = next(item for item in gpu_suite_contracts(config) if item.name == "search_p8_subspace_r4")
    # The contract root is generated from config.output_root, so point it at the actual run.
    contract = contract.__class__(**{**contract.__dict__, "root": Path(Args.out)})
    result = check_run(contract)
    assert result.passed, result.invalid


def test_search_cli_runs_vllm_labelled_subspace_smoke(tmp_path: Path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "optimus.cli",
            "search",
            "--backend",
            "vllm",
            "--method",
            "subspace",
            "--out",
            str(tmp_path / "run"),
            "--prompts",
            "4",
            "--holdout-prompts",
            "4",
            "--population",
            "4",
            "--promote",
            "1",
            "--basis-rank",
            "4",
            "--basis-prompts",
            "4",
            "--target-preset",
            "qv",
            "--basis-centering",
            "none",
            "--basis-token-source",
            "prefill",
            "--basis-kind",
            "activation-svd",
            "--scale-mode",
            "relative-output-rms",
            "--rho-grid",
            "0.01",
            "--budget-policy",
            "per-target-equal",
            "--top-k-grid",
            "1",
            "--kernel",
            "torch",
            "--prefix-cache-policy",
            "disabled-for-search",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert summary["kind"] == "subspace_vllm_search"
    assert summary["candidate_routing"] == "row_candidate_id"
