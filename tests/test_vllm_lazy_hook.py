from dataclasses import replace

import torch

from optimus.backends.vllm_lazy_hook import HookTarget, LazyHookRuntime
from optimus.backends.vllm_lazy_hook import _ensemble_input_rows
from optimus.backends.vllm_lora_hook import FusedQKVSpec, LazyLoraHookRuntime
from optimus.core.perturbations import PerturbationSpec
from optimus.modeling.subspace_lora import subspace_lora_tensors
from optimus.modeling.noise import lora_noise_tensors
from optimus.search.ensemble import majority_vote_evaluation
from optimus.subspace import SubspaceCandidate
from optimus.tasks.countdown import CountdownExample
from scripts.eval_vllm_lazy_k1 import _load_betas


def test_vllm_lazy_hook_ensemble_rows_normalize_candidate_id():
    example = CountdownExample(7, (3, 7, 8, 8), 24)
    rows = [
        {
            "split": "holdout",
            "candidate_id": "__base__",
            "example_id": example.id,
            "numbers": list(example.numbers),
            "target": example.target,
            "text": "<answer>8*(7-3)-8</answer>",
        },
        {
            "split": "screen",
            "candidate_id": "screen-only",
            "example_id": example.id,
            "numbers": list(example.numbers),
            "target": example.target,
            "text": "<answer>8*(7-3)-8</answer>",
        },
        {
            "split": "holdout",
            "candidate_id": "cand-invalid",
            "example_id": example.id,
            "numbers": list(example.numbers),
            "target": example.target,
            "text": "<answer>24</answer>",
        },
        {
            "split": "holdout",
            "candidate_id": "cand-correct",
            "example_id": example.id,
            "numbers": list(example.numbers),
            "target": example.target,
            "text": "<answer>8*(7-3)-8</answer>",
        },
    ]

    ensemble_rows = _ensemble_input_rows(rows, split="holdout")

    assert [row["candidate"] for row in ensemble_rows] == ["cand-invalid", "cand-correct"]
    scores, per_prompt = majority_vote_evaluation(
        ["cand-invalid", "cand-correct"],
        ensemble_rows,
        [example],
        [1, 2],
    )
    assert scores[0]["exact_mean"] == 0.0
    assert scores[1]["exact_mean"] == 1.0
    assert per_prompt[-1]["valid_vote_count"] == 1


def _candidate(candidate_id: str, seed: int) -> SubspaceCandidate:
    return SubspaceCandidate(
        candidate_id=candidate_id,
        direction_seed=seed,
        sign="+",
        basis_hash="basis",
        target_set_hash="targets",
        scale_mode="relative-output-rms",
        rho_or_sigma_w=0.1,
        budget_policy="per-target-equal",
        budget_hash="budget",
        runtime_dtype="fp32",
        radius_index=0,
        target_preset="qv",
        basis_rank=2,
        shard_id="single",
        shard_population_start=0,
        shard_population_end=2,
        worker_id="worker",
        device_id="cpu",
        prompt_scoring_config_hash="prompt",
    )


def test_vllm_lazy_hook_row_candidate_batch_matches_serial_deltas():
    target = HookTarget(
        module_name="model.layers.0.self_attn.q_proj",
        target_id="layer_0.self_attn.q_proj",
        site_id="layer_0.attn_in",
        layer_index=0,
        block_path="model.layers.0",
        suffix="q_proj",
        module=torch.nn.Linear(2, 3),
        input_dim=2,
        output_dim=3,
    )
    runtime = LazyHookRuntime([target])
    runtime.basis_by_site[target.site_id] = torch.eye(2)
    runtime.beta_by_target[target.target_id] = 0.25
    x = torch.tensor([[1.0, 0.0], [2.0, 1.0], [0.5, 1.5], [1.0, -1.0]])
    y = torch.zeros((4, 3))
    cand_a = _candidate("a", 11)
    cand_b = _candidate("b", 29)

    runtime.set_candidate(cand_a)
    serial_a = runtime.delta(target, x[:2], y[:2])
    runtime.set_candidate(cand_b)
    serial_b = runtime.delta(target, x[2:], y[2:])

    runtime.set_candidate_batch({"10": cand_a, "11": cand_b})
    runtime.update_row_candidates(["10", "11"], [0, 2, 4])
    batched = runtime.delta(target, x, y)

    assert torch.allclose(batched[:2], serial_a)
    assert torch.allclose(batched[2:], serial_b)


def test_vllm_lazy_hook_zero_radius_counts_rows_and_returns_zero_delta():
    target = HookTarget(
        module_name="model.layers.0.self_attn.q_proj",
        target_id="layer_0.self_attn.q_proj",
        site_id="layer_0.attn_in",
        layer_index=0,
        block_path="model.layers.0",
        suffix="q_proj",
        module=torch.nn.Linear(2, 3),
        input_dim=2,
        output_dim=3,
    )
    runtime = LazyHookRuntime([target])
    runtime.basis_by_site[target.site_id] = torch.eye(2)
    runtime.beta_by_target[target.target_id] = 0.0
    runtime.set_candidate(_candidate("zero", 11))
    x = torch.tensor([[1.0, 0.0], [2.0, 1.0]])
    y = torch.randn((2, 3))

    delta = runtime.delta(target, x, y)

    assert torch.equal(delta, torch.zeros_like(y))
    assert runtime.delta_rows == 2
    assert runtime.delta_calls == 1


def test_vllm_lora_hook_matches_canonical_lora_delta():
    target = HookTarget(
        module_name="model.layers.0.self_attn.q_proj",
        target_id="layer_0.self_attn.q_proj",
        site_id="layer_0.attn_in",
        layer_index=0,
        block_path="model.layers.0",
        suffix="q_proj",
        module=torch.nn.Linear(4, 3),
        input_dim=4,
        output_dim=3,
    )
    rank = 2
    candidate = PerturbationSpec(
        "isotropic",
        123,
        0.0075,
        1,
        method="lora",
        rank=rank,
        targets=("q_proj", "v_proj"),
    )
    runtime = LazyLoraHookRuntime([target], rank=rank, adapter_dtype="float32")
    runtime.set_candidate(candidate)
    x = torch.tensor([[1.0, -2.0, 0.5, 3.0], [0.25, 1.5, -1.0, 2.0]])
    y = torch.zeros((2, 3))

    delta = runtime.delta(target, x, y)

    a, b = lora_noise_tensors(target.module_name, (rank, 4), (3, rank), candidate, rank, state_key=target.module_name)
    expected = (x @ a.T) @ b.T
    assert torch.allclose(delta, expected)
    assert runtime.delta_rows == 2
    assert runtime.delta_calls == 1


def test_vllm_lora_hook_row_candidate_batch_matches_serial_deltas():
    target = HookTarget(
        module_name="model.layers.0.self_attn.v_proj",
        target_id="layer_0.self_attn.v_proj",
        site_id="layer_0.attn_in",
        layer_index=0,
        block_path="model.layers.0",
        suffix="v_proj",
        module=torch.nn.Linear(3, 2),
        input_dim=3,
        output_dim=2,
    )
    rank = 2
    cand_a = PerturbationSpec("isotropic", 11, 0.01, 1, method="lora", rank=rank, targets=("q_proj", "v_proj"))
    cand_b = PerturbationSpec("isotropic", 29, 0.01, -1, method="lora", rank=rank, targets=("q_proj", "v_proj"))
    runtime = LazyLoraHookRuntime([target], rank=rank, adapter_dtype="float32")
    x = torch.tensor([[1.0, 0.0, 2.0], [2.0, 1.0, -1.0], [0.5, 1.5, 1.0], [1.0, -1.0, 0.0]])
    y = torch.zeros((4, 2))

    runtime.set_candidate(cand_a)
    serial_a = runtime.delta(target, x[:2], y[:2])
    runtime.set_candidate(cand_b)
    serial_b = runtime.delta(target, x[2:], y[2:])

    runtime.set_candidate_batch({"10": cand_a, "11": cand_b})
    runtime.update_row_candidates(["10", "11"], [0, 2, 4])
    batched = runtime.delta(target, x, y)

    assert torch.allclose(batched[:2], serial_a)
    assert torch.allclose(batched[2:], serial_b)


def test_vllm_lora_hook_input_order_candidate_batch_matches_serial_deltas():
    target = HookTarget(
        module_name="model.layers.0.self_attn.v_proj",
        target_id="layer_0.self_attn.v_proj",
        site_id="layer_0.attn_in",
        layer_index=0,
        block_path="model.layers.0",
        suffix="v_proj",
        module=torch.nn.Linear(3, 2),
        input_dim=3,
        output_dim=2,
    )
    rank = 2
    cand_a = PerturbationSpec("isotropic", 11, 0.01, 1, method="lora", rank=rank, targets=("q_proj", "v_proj"))
    cand_b = PerturbationSpec("isotropic", 29, 0.01, -1, method="lora", rank=rank, targets=("q_proj", "v_proj"))
    runtime = LazyLoraHookRuntime([target], rank=rank, adapter_dtype="float32")
    x = torch.tensor([[1.0, 0.0, 2.0], [2.0, 1.0, -1.0], [0.5, 1.5, 1.0], [1.0, -1.0, 0.0]])
    y = torch.zeros((4, 2))

    runtime.set_candidate(cand_a)
    serial_a = runtime.delta(target, x[:2], y[:2])
    runtime.set_candidate(cand_b)
    serial_b = runtime.delta(target, x[2:], y[2:])

    runtime.set_candidate_batch_by_order([cand_a, cand_b], prompt_count=1)
    runtime.update_row_candidates(["any-a", "any-b"], [0, 2, 4])
    batched = runtime.delta(target, x, y)

    assert torch.allclose(batched[:2], serial_a)
    assert torch.allclose(batched[2:], serial_b)


def test_vllm_lora_hook_fused_qkv_injects_requested_slices():
    target = HookTarget(
        module_name="model.layers.0.self_attn.qkv_proj",
        target_id="layer_0.self_attn.qkv_proj",
        site_id="layer_0.attn_in",
        layer_index=0,
        block_path="model.layers.0",
        suffix="qkv_proj",
        module=torch.nn.Linear(4, 7),
        input_dim=4,
        output_dim=7,
    )
    rank = 2
    candidate = PerturbationSpec(
        "isotropic",
        123,
        0.0075,
        1,
        method="lora",
        rank=rank,
        targets=("q_proj", "v_proj"),
    )
    runtime = LazyLoraHookRuntime([target], rank=rank, adapter_dtype="float32", fused_qkv=FusedQKVSpec(q_out=3, kv_out=2))
    runtime.set_candidate(candidate)
    x = torch.tensor([[1.0, -2.0, 0.5, 3.0], [0.25, 1.5, -1.0, 2.0]])
    y = torch.zeros((2, 7))

    delta = runtime.delta(target, x, y)

    q_a, q_b = lora_noise_tensors("model.layers.0.self_attn.q_proj", (rank, 4), (3, rank), candidate, rank)
    v_a, v_b = lora_noise_tensors("model.layers.0.self_attn.v_proj", (rank, 4), (2, rank), candidate, rank)
    expected = torch.zeros_like(y)
    expected[:, :3] = (x @ q_a.T) @ q_b.T
    expected[:, 5:7] = (x @ v_a.T) @ v_b.T

    assert torch.allclose(delta, expected)
    assert torch.equal(delta[:, 3:5], torch.zeros_like(delta[:, 3:5]))


def test_vllm_lora_hook_preload_preserves_factor_cache_across_candidates():
    target = HookTarget(
        module_name="model.layers.0.self_attn.q_proj",
        target_id="layer_0.self_attn.q_proj",
        site_id="layer_0.attn_in",
        layer_index=0,
        block_path="model.layers.0",
        suffix="q_proj",
        module=torch.nn.Linear(4, 3),
        input_dim=4,
        output_dim=3,
    )
    rank = 2
    candidate = PerturbationSpec("isotropic", 123, 0.0075, 1, method="lora", rank=rank, targets=("q_proj",))
    runtime = LazyLoraHookRuntime([target], rank=rank, adapter_dtype="float32", preserve_factor_cache=True)

    runtime.preload_candidate(candidate)
    cached = len(runtime._factor_cache)
    runtime.set_candidate(candidate)
    x = torch.randn((2, 4))
    y = torch.zeros((2, 3))
    runtime.delta(target, x, y)

    assert cached == 1
    assert len(runtime._factor_cache) == cached


class _TinyQwenConfig:
    model_type = "qwen3"
    hidden_size = 4
    intermediate_size = 8
    num_hidden_layers = 1
    num_attention_heads = 2
    num_key_value_heads = 1
    head_dim = 2


def _subspace_candidate_for_adapter() -> SubspaceCandidate:
    return SubspaceCandidate(
        candidate_id="seed11:+:r2:rho0.4",
        direction_seed=11,
        sign="+",
        basis_hash="basis",
        target_set_hash="targets",
        scale_mode="relative-output-rms",
        rho_or_sigma_w=0.4,
        budget_policy="per-target-equal",
        budget_hash="budget",
        runtime_dtype="bf16",
        radius_index=0,
        target_preset="qv",
        basis_rank=2,
        shard_id="single",
        shard_population_start=0,
        shard_population_end=1,
        worker_id="test",
        device_id="cpu",
        prompt_scoring_config_hash="prompt",
        rng_version="torch_generator_field_v1",
    )


def test_subspace_adapter_bridge_materializes_fused_qkv_exact_slices():
    basis = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    state_payload = {"basis_tensors": {"basis/layer_0.attn_in": basis}}
    state_summary = {
        "activation_sites": [
            {
                "site_id": "layer_0.attn_in",
                "basis_tensor_key": "basis/layer_0.attn_in",
            }
        ]
    }
    source_summary = {
        "resolved_target_scales": [
            {
                "target_id": "layer_0.self_attn.qkv_proj",
                "beta_t_by_radius": {"0.4": 0.25},
            }
        ]
    }
    candidate = _subspace_candidate_for_adapter()

    tensors = subspace_lora_tensors(
        config=_TinyQwenConfig(),
        state_payload=state_payload,
        state_summary=state_summary,
        source_summary=source_summary,
        candidate=candidate,
        targets=["q_proj", "k_proj", "v_proj"],
        policy="fused-qkv-exact",
        tensor_dtype="float32",
    )

    q_a = tensors["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"]
    q_b = tensors["base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"]
    k_a = tensors["base_model.model.model.layers.0.self_attn.k_proj.lora_A.weight"]
    k_b = tensors["base_model.model.model.layers.0.self_attn.k_proj.lora_B.weight"]
    v_b = tensors["base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight"]

    assert torch.equal(q_a, basis)
    assert torch.equal(k_a, basis)
    assert q_b.shape == (4, 2)
    assert k_b.shape == (2, 2)
    assert v_b.shape == (2, 2)


def test_subspace_adapter_bridge_target_split_uses_requested_targets_only():
    basis = torch.eye(4)[:2]
    state_payload = {"basis_tensors": {"basis/layer_0.attn_in": basis}}
    state_summary = {"activation_sites": [{"site_id": "layer_0.attn_in", "basis_tensor_key": "basis/layer_0.attn_in"}]}
    source_summary = {
        "resolved_target_scales": [
            {
                "target_id": "layer_0.self_attn.qkv_proj",
                "beta_t_by_radius": {"0.4": 0.25},
            }
        ]
    }

    tensors = subspace_lora_tensors(
        config=_TinyQwenConfig(),
        state_payload=state_payload,
        state_summary=state_summary,
        source_summary=source_summary,
        candidate=_subspace_candidate_for_adapter(),
        targets=["q_proj", "v_proj"],
        policy="target-split",
        tensor_dtype="float32",
    )

    assert "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight" in tensors
    assert "base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight" in tensors
    assert not any(".k_proj." in key for key in tensors)


def test_vllm_lazy_replay_expands_fused_qkv_betas_and_scales():
    source_summary = {
        "resolved_target_scales": [
            {
                "target_id": "layer_0.self_attn.qkv_proj",
                "beta_t_by_radius": {"0.4": 0.25},
            }
        ]
    }

    betas = _load_betas(source_summary, radius=0.4, scale_multiplier=2.0)

    assert betas["layer_0.self_attn.qkv_proj"] == 0.5
    assert betas["layer_0.self_attn.q_proj"] == 0.5
    assert betas["layer_0.self_attn.k_proj"] == 0.5
    assert betas["layer_0.self_attn.v_proj"] == 0.5


def test_vllm_lazy_replay_matches_target_split_subspace_adapter_rank_and_scale():
    basis = torch.eye(4)[:3].contiguous()
    state_payload = {"basis_tensors": {"basis/layer_0.attn_in": basis}}
    state_summary = {"activation_sites": [{"site_id": "layer_0.attn_in", "basis_tensor_key": "basis/layer_0.attn_in"}]}
    source_summary = {
        "resolved_target_scales": [
            {
                "target_id": "layer_0.self_attn.qkv_proj",
                "beta_t_by_radius": {"0.4": 0.25},
            }
        ]
    }
    candidate = replace(_subspace_candidate_for_adapter(), basis_rank=3)
    target = HookTarget(
        module_name="model.layers.0.self_attn.q_proj",
        target_id="layer_0.self_attn.q_proj",
        site_id="layer_0.attn_in",
        layer_index=0,
        block_path="model.layers.0",
        suffix="q_proj",
        module=torch.nn.Linear(4, 4),
        input_dim=4,
        output_dim=4,
    )
    runtime = LazyHookRuntime([target])
    runtime.basis_by_site[target.site_id] = basis[:2].contiguous()
    runtime.beta_by_target.update(_load_betas(source_summary, radius=0.4, scale_multiplier=2.0))
    runtime.set_candidate(candidate)
    x = torch.tensor([[1.0, -2.0, 0.5, 3.0], [0.25, 1.5, -1.0, 2.0]])
    y = torch.zeros((2, 4))

    delta = runtime.delta(target, x, y)
    tensors = subspace_lora_tensors(
        config=_TinyQwenConfig(),
        state_payload=state_payload,
        state_summary=state_summary,
        source_summary=source_summary,
        candidate=candidate,
        targets=["q_proj"],
        policy="target-split",
        tensor_dtype="float32",
        adapter_rank=2,
        scale_multiplier=2.0,
    )
    a = tensors["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"]
    b = tensors["base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"]
    expected = (x @ a.T) @ b.T

    assert torch.allclose(delta, expected)
