from dataclasses import replace
from types import SimpleNamespace

import torch
import pytest

from optimus.backends.vllm_lazy_hook import HookTarget, LazyHookRuntime
from optimus.backends.vllm_lazy_hook import _ensemble_input_rows
from scripts import eval_vllm_lazy_k1 as lazy_k1
from optimus.backends.vllm_lora_hook import FusedQKVSpec, LazyLoraHookRuntime
from optimus.core.perturbations import PerturbationSpec
from optimus.modeling.subspace_lora import subspace_lora_tensors
from optimus.modeling.noise import lora_noise_tensors
from optimus.search.ensemble import majority_vote_evaluation
from optimus.subspace import SubspaceCandidate, counter_gaussian_v1, gaussian_hash_v1, stable_u32
from optimus.tasks.countdown import CountdownExample
from scripts.eval_vllm_lazy_k1 import _filter_targets, _load_betas


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


def test_vllm_lazy_hook_internal_policy_env_validation(monkeypatch):
    monkeypatch.setenv("OPTIMUS_LAZY_FIELD_POLICY", "fused-qkv-exact")
    monkeypatch.setenv("OPTIMUS_LAZY_QKV_KERNEL_POLICY", "packed-qkv")

    runtime = LazyHookRuntime([])

    assert runtime.field_policy == "fused-qkv-exact"
    assert runtime.qkv_kernel_policy == "packed-qkv"

    monkeypatch.setenv("OPTIMUS_LAZY_QKV_KERNEL_POLICY", "bad-policy")
    with pytest.raises(ValueError, match="OPTIMUS_LAZY_QKV_KERNEL_POLICY"):
        LazyHookRuntime([])


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


def test_vllm_lazy_hook_input_order_candidate_batch_matches_serial_deltas():
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

    runtime.set_candidate_batch_by_order([cand_a, cand_b], prompt_count=1)
    runtime.update_row_candidates(["opaque-a", "opaque-b"], [0, 2, 4])
    batched = runtime.delta(target, x, y)

    assert torch.allclose(batched[:2], serial_a)
    assert torch.allclose(batched[2:], serial_b)


def test_vllm_lazy_hook_numeric_request_ids_route_candidate_batch_when_scheduler_reorders():
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

    runtime.set_candidate_batch_by_order([cand_a, cand_b], prompt_count=2)
    runtime.update_row_candidates(["101-any", "100-any", "102-any"], [0, 1, 2, 4])
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


def test_vllm_lazy_hook_reuses_vllm_kernel_factor_stacks():
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
    candidates = [_candidate("a", 11), _candidate("b", 29)]
    basis = runtime.basis_for(target.site_id, device=torch.device("cpu"), dtype=torch.float32)
    assert basis is not None

    a_first = runtime._vllm_a_stack(
        target,
        basis,
        num_loras=2,
        input_dim=2,
        rank=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    a_second = runtime._vllm_a_stack(
        target,
        basis,
        num_loras=2,
        input_dim=2,
        rank=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    b_first = runtime._vllm_b_stack(
        target,
        candidates,
        output_dim=3,
        rank=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        beta=0.25,
        target_id=target.target_id,
    )
    b_second = runtime._vllm_b_stack(
        target,
        candidates,
        output_dim=3,
        rank=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        beta=0.25,
        target_id=target.target_id,
    )

    assert a_first.data_ptr() == a_second.data_ptr()
    assert b_first.data_ptr() == b_second.data_ptr()
    runtime.set_candidate(None)
    assert runtime._vllm_a_stack_cache == {}
    assert runtime._vllm_b_stack_cache == {}


def test_vllm_lazy_hook_reuses_prepared_vllm_kernel_metadata():
    target = HookTarget(
        module_name="model.layers.0.self_attn.q_proj",
        target_id="layer_0.self_attn.q_proj",
        site_id="layer_0.attn_in",
        layer_index=0,
        block_path="model.layers.0",
        suffix="q_proj",
        module=torch.nn.Linear(2, 3),
    )
    runtime = LazyHookRuntime([target])
    mapping = torch.tensor([0, 0, 1, 1], dtype=torch.int16)
    cand_a = _candidate("a", 11)
    cand_b = _candidate("b", 29)

    class FakeMeta:
        def __init__(self) -> None:
            self.calls = 0

        def prepare_tensors(self, token_mapping: torch.Tensor) -> None:
            self.calls += 1
            assert torch.equal(token_mapping.cpu(), mapping.to(dtype=torch.int32))

    meta = FakeMeta()
    token_mapping, mapping_key = runtime._token_mapping_for(mapping, device=torch.device("cpu"), rows=4, num_loras=2)
    meta_key = ("cpu", 2, 4)

    runtime._prepare_vllm_meta(meta, token_mapping, meta_key=meta_key, mapping_key=mapping_key)
    runtime._prepare_vllm_meta(meta, token_mapping, meta_key=meta_key, mapping_key=mapping_key)

    assert meta.calls == 1
    token_mapping_again, mapping_key_again = runtime._token_mapping_for(mapping, device=torch.device("cpu"), rows=4, num_loras=2)
    assert token_mapping.data_ptr() == token_mapping_again.data_ptr()
    assert mapping_key == mapping_key_again
    runtime.set_candidate_batch_by_order([cand_a, cand_b], prompt_count=1)
    runtime.update_row_candidates(["0", "1"], [0, 2, 4])
    assert runtime._mapping_cache_key(device=torch.device("cpu"), rows=4, num_loras=2) != mapping_key


def test_vllm_lazy_hook_vllm_kernel_backend_fails_closed_on_cpu():
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
    runtime.delta_backend = "vllm-lora-kernel"
    runtime.compute_dtype_policy = "bfloat16"
    runtime.basis_by_site[target.site_id] = torch.eye(2)
    runtime.beta_by_target[target.target_id] = 0.25
    runtime.set_candidate(_candidate("a", 11))

    with pytest.raises(RuntimeError, match="requires CUDA tensors"):
        runtime.delta(target, torch.ones((2, 2)), torch.zeros((2, 3)))


def test_vllm_lazy_hook_triton_backend_fails_closed_on_cpu():
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
    runtime.delta_backend = "triton"
    runtime.compute_dtype_policy = "bfloat16"
    runtime.basis_by_site[target.site_id] = torch.eye(2)
    runtime.beta_by_target[target.target_id] = 0.25
    runtime.set_candidate(_candidate("a", 11))

    with pytest.raises(RuntimeError, match="triton|CUDA"):
        runtime.delta(target, torch.ones((2, 2)), torch.zeros((2, 3)))


def test_vllm_lazy_hook_triton_counter_backend_fails_closed_on_cpu():
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
    runtime.delta_backend = "triton-counter"
    runtime.compute_dtype_policy = "bfloat16"
    runtime.basis_by_site[target.site_id] = torch.eye(2)
    runtime.beta_by_target[target.target_id] = 0.25
    runtime.set_candidate(replace(_candidate("a", 11), rng_version="counter_gaussian_v1"))

    with pytest.raises(RuntimeError, match="triton|CUDA"):
        runtime.delta(target, torch.ones((2, 2)), torch.zeros((2, 3)))


def test_vllm_lazy_hook_triton_counter_inplace_backend_fails_closed_on_cpu():
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
    runtime.delta_backend = "triton-counter-inplace"
    runtime.compute_dtype_policy = "bfloat16"
    runtime.basis_by_site[target.site_id] = torch.eye(2)
    runtime.beta_by_target[target.target_id] = 0.25
    runtime.set_candidate(replace(_candidate("a", 11), rng_version="counter_gaussian_v1"))

    with pytest.raises(RuntimeError, match="triton|CUDA"):
        runtime.delta(target, torch.ones((2, 2)), torch.zeros((2, 3)))


def test_vllm_lazy_hook_triton_counter_rejects_non_counter_rng_on_cpu():
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
    runtime.delta_backend = "triton-counter"
    runtime.compute_dtype_policy = "bfloat16"
    runtime.basis_by_site[target.site_id] = torch.eye(2)
    runtime.beta_by_target[target.target_id] = 0.25
    runtime.set_candidate(replace(_candidate("a", 11), rng_version="gaussian_hash_v1"))

    with pytest.raises(RuntimeError, match="counter_gaussian_v1"):
        runtime.delta(target, torch.ones((2, 2)), torch.zeros((2, 3)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_vllm_lazy_hook_triton_counter_cuda_matches_torch_counter_rng():
    target = HookTarget(
        module_name="model.layers.0.self_attn.q_proj",
        target_id="layer_0.self_attn.q_proj",
        site_id="layer_0.attn_in",
        layer_index=0,
        block_path="model.layers.0",
        suffix="q_proj",
        module=torch.nn.Linear(16, 20),
        input_dim=16,
        output_dim=20,
    )
    torch.manual_seed(0)
    basis = torch.randn((4, 16), dtype=torch.float32)
    x = torch.randn((7, 16), device="cuda", dtype=torch.float32)
    y = torch.zeros((7, 20), device="cuda", dtype=torch.float32)
    cand_a = replace(_candidate("a", 11), rng_version="counter_gaussian_v1", basis_rank=4, sign="+")
    cand_b = replace(_candidate("b", 29), rng_version="counter_gaussian_v1", basis_rank=4, sign="-")

    torch_runtime = LazyHookRuntime([target])
    torch_runtime.delta_backend = "torch"
    torch_runtime.compute_dtype_policy = "float32"
    torch_runtime.basis_by_site[target.site_id] = basis
    torch_runtime.beta_by_target[target.target_id] = 0.125
    torch_runtime.set_candidate_batch_by_order([cand_a, cand_b], prompt_count=1)
    torch_runtime.update_row_candidates(["0", "1"], [0, 3, 7])

    triton_runtime = LazyHookRuntime([target])
    triton_runtime.delta_backend = "triton-counter"
    triton_runtime.compute_dtype_policy = "float32"
    triton_runtime.basis_by_site[target.site_id] = basis
    triton_runtime.beta_by_target[target.target_id] = 0.125
    triton_runtime.set_candidate_batch_by_order([cand_a, cand_b], prompt_count=1)
    triton_runtime.update_row_candidates(["0", "1"], [0, 3, 7])

    expected = torch_runtime.delta(target, x, y)
    actual = triton_runtime.delta(target, x, y)

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    assert triton_runtime.delta_rows == 7
    assert triton_runtime.delta_calls == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_vllm_lazy_hook_triton_counter_inplace_cuda_matches_torch_counter_rng():
    target = HookTarget(
        module_name="model.layers.0.self_attn.q_proj",
        target_id="layer_0.self_attn.q_proj",
        site_id="layer_0.attn_in",
        layer_index=0,
        block_path="model.layers.0",
        suffix="q_proj",
        module=torch.nn.Linear(16, 20),
        input_dim=16,
        output_dim=20,
    )
    torch.manual_seed(2)
    basis = torch.randn((4, 16), dtype=torch.float32)
    x = torch.randn((7, 16), device="cuda", dtype=torch.float32)
    base = torch.randn((7, 20), device="cuda", dtype=torch.float32)
    cand_a = replace(_candidate("a", 11), rng_version="counter_gaussian_v1", basis_rank=4, sign="+")
    cand_b = replace(_candidate("b", 29), rng_version="counter_gaussian_v1", basis_rank=4, sign="-")

    torch_runtime = LazyHookRuntime([target])
    torch_runtime.delta_backend = "torch"
    torch_runtime.compute_dtype_policy = "float32"
    torch_runtime.basis_by_site[target.site_id] = basis
    torch_runtime.beta_by_target[target.target_id] = 0.125
    torch_runtime.set_candidate_batch_by_order([cand_a, cand_b], prompt_count=1)
    torch_runtime.update_row_candidates(["0", "1"], [0, 3, 7])

    inplace_runtime = LazyHookRuntime([target])
    inplace_runtime.delta_backend = "triton-counter-inplace"
    inplace_runtime.compute_dtype_policy = "float32"
    inplace_runtime.basis_by_site[target.site_id] = basis
    inplace_runtime.beta_by_target[target.target_id] = 0.125
    inplace_runtime.set_candidate_batch_by_order([cand_a, cand_b], prompt_count=1)
    inplace_runtime.update_row_candidates(["0", "1"], [0, 3, 7])

    expected = base + torch_runtime.delta(target, x, torch.zeros_like(base))
    actual_base = base.clone()
    actual = inplace_runtime.delta(target, x, actual_base)

    assert inplace_runtime._last_delta_is_output
    assert actual.data_ptr() == actual_base.data_ptr()
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_vllm_lazy_hook_triton_counter_cuda_matches_fused_qkv_exact_offsets():
    target = HookTarget(
        module_name="model.layers.0.self_attn.qkv_proj",
        target_id="layer_0.self_attn.qkv_proj",
        site_id="layer_0.attn_in",
        layer_index=0,
        block_path="model.layers.0",
        suffix="qkv_proj",
        module=torch.nn.Linear(12, 20),
        input_dim=12,
        output_dim=20,
        fused_qkv_slices=("q_proj", "v_proj"),
        fused_q_out=8,
        fused_kv_out=6,
    )
    torch.manual_seed(1)
    basis = torch.randn((4, 12), dtype=torch.float32)
    x = torch.randn((5, 12), device="cuda", dtype=torch.float32)
    y = torch.zeros((5, 20), device="cuda", dtype=torch.float32)
    cand = replace(_candidate("a", 41), rng_version="counter_gaussian_v1", basis_rank=4, sign="-")
    beta_by_target = {
        "layer_0.self_attn.q_proj": 0.03,
        "layer_0.self_attn.v_proj": 0.07,
    }

    torch_runtime = LazyHookRuntime([target])
    torch_runtime.delta_backend = "torch"
    torch_runtime.field_policy = "fused-qkv-exact"
    torch_runtime.compute_dtype_policy = "float32"
    torch_runtime.basis_by_site[target.site_id] = basis
    torch_runtime.beta_by_target.update(beta_by_target)
    torch_runtime.set_candidate(cand)

    triton_runtime = LazyHookRuntime([target])
    triton_runtime.delta_backend = "triton-counter"
    triton_runtime.field_policy = "fused-qkv-exact"
    triton_runtime.compute_dtype_policy = "float32"
    triton_runtime.basis_by_site[target.site_id] = basis
    triton_runtime.beta_by_target.update(beta_by_target)
    triton_runtime.set_candidate(cand)

    expected = torch_runtime.delta(target, x, y)
    actual = triton_runtime.delta(target, x, y)

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    assert torch.count_nonzero(actual[:, 8:14]) == 0
    assert torch.count_nonzero(actual[:, :8]) > 0
    assert torch.count_nonzero(actual[:, 14:]) > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_vllm_lazy_hook_triton_counter_inplace_cuda_matches_fused_qkv_exact_offsets():
    target = HookTarget(
        module_name="model.layers.0.self_attn.qkv_proj",
        target_id="layer_0.self_attn.qkv_proj",
        site_id="layer_0.attn_in",
        layer_index=0,
        block_path="model.layers.0",
        suffix="qkv_proj",
        module=torch.nn.Linear(12, 20),
        input_dim=12,
        output_dim=20,
        fused_qkv_slices=("q_proj", "v_proj"),
        fused_q_out=8,
        fused_kv_out=6,
    )
    torch.manual_seed(3)
    basis = torch.randn((4, 12), dtype=torch.float32)
    x = torch.randn((5, 12), device="cuda", dtype=torch.float32)
    base = torch.randn((5, 20), device="cuda", dtype=torch.float32)
    cand = replace(_candidate("a", 41), rng_version="counter_gaussian_v1", basis_rank=4, sign="-")
    beta_by_target = {
        "layer_0.self_attn.q_proj": 0.03,
        "layer_0.self_attn.v_proj": 0.07,
    }

    torch_runtime = LazyHookRuntime([target])
    torch_runtime.delta_backend = "torch"
    torch_runtime.field_policy = "fused-qkv-exact"
    torch_runtime.compute_dtype_policy = "float32"
    torch_runtime.basis_by_site[target.site_id] = basis
    torch_runtime.beta_by_target.update(beta_by_target)
    torch_runtime.set_candidate(cand)

    inplace_runtime = LazyHookRuntime([target])
    inplace_runtime.delta_backend = "triton-counter-inplace"
    inplace_runtime.field_policy = "fused-qkv-exact"
    inplace_runtime.compute_dtype_policy = "float32"
    inplace_runtime.basis_by_site[target.site_id] = basis
    inplace_runtime.beta_by_target.update(beta_by_target)
    inplace_runtime.set_candidate(cand)

    expected = base + torch_runtime.delta(target, x, torch.zeros_like(base))
    actual_base = base.clone()
    actual = inplace_runtime.delta(target, x, actual_base)

    assert inplace_runtime._last_delta_is_output
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    assert torch.equal(actual[:, 8:14], base[:, 8:14])
    assert torch.count_nonzero(actual[:, :8] - base[:, :8]) > 0
    assert torch.count_nonzero(actual[:, 14:] - base[:, 14:]) > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_vllm_lazy_hook_triton_counter_inplace_cuda_matches_target_split_qkv_offsets():
    target = HookTarget(
        module_name="model.layers.0.self_attn.qkv_proj",
        target_id="layer_0.self_attn.qkv_proj",
        site_id="layer_0.attn_in",
        layer_index=0,
        block_path="model.layers.0",
        suffix="qkv_proj",
        module=torch.nn.Linear(12, 20),
        input_dim=12,
        output_dim=20,
        fused_qkv_slices=("q_proj", "v_proj"),
        fused_q_out=8,
        fused_kv_out=6,
    )
    torch.manual_seed(4)
    basis = torch.randn((4, 12), dtype=torch.float32)
    x = torch.randn((5, 12), device="cuda", dtype=torch.float32)
    base = torch.randn((5, 20), device="cuda", dtype=torch.float32)
    cand = replace(_candidate("a", 43), rng_version="counter_gaussian_v1", basis_rank=4, sign="+")
    beta_by_target = {
        "layer_0.self_attn.q_proj": 0.05,
        "layer_0.self_attn.v_proj": 0.09,
    }

    torch_runtime = LazyHookRuntime([target])
    torch_runtime.delta_backend = "torch"
    torch_runtime.field_policy = "target-split"
    torch_runtime.compute_dtype_policy = "float32"
    torch_runtime.basis_by_site[target.site_id] = basis
    torch_runtime.beta_by_target.update(beta_by_target)
    torch_runtime.set_candidate(cand)

    inplace_runtime = LazyHookRuntime([target])
    inplace_runtime.delta_backend = "triton-counter-inplace"
    inplace_runtime.field_policy = "target-split"
    inplace_runtime.compute_dtype_policy = "float32"
    inplace_runtime.basis_by_site[target.site_id] = basis
    inplace_runtime.beta_by_target.update(beta_by_target)
    inplace_runtime.set_candidate(cand)

    expected = base + torch_runtime.delta(target, x, torch.zeros_like(base))
    actual_base = base.clone()
    actual = inplace_runtime.delta(target, x, actual_base)

    assert inplace_runtime._last_delta_is_output
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    assert torch.equal(actual[:, 8:14], base[:, 8:14])
    assert torch.count_nonzero(actual[:, :8] - base[:, :8]) > 0
    assert torch.count_nonzero(actual[:, 14:] - base[:, 14:]) > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_vllm_lazy_hook_triton_counter_inplace_packed_qv_matches_split_qv_offsets():
    target = HookTarget(
        module_name="model.layers.0.self_attn.qkv_proj",
        target_id="layer_0.self_attn.qkv_proj",
        site_id="layer_0.attn_in",
        layer_index=0,
        block_path="model.layers.0",
        suffix="qkv_proj",
        module=torch.nn.Linear(12, 20),
        input_dim=12,
        output_dim=20,
        fused_qkv_slices=("q_proj", "v_proj"),
        fused_q_out=8,
        fused_kv_out=6,
    )
    torch.manual_seed(5)
    basis = torch.randn((4, 12), dtype=torch.float32)
    x = torch.randn((7, 12), device="cuda", dtype=torch.float32)
    base = torch.randn((7, 20), device="cuda", dtype=torch.float32)
    cand_a = replace(_candidate("a", 47), rng_version="counter_gaussian_v1", basis_rank=4, sign="+")
    cand_b = replace(_candidate("b", 49), rng_version="counter_gaussian_v1", basis_rank=4, sign="-")
    beta_by_target = {
        "layer_0.self_attn.q_proj": 0.05,
        "layer_0.self_attn.v_proj": 0.09,
    }

    split_runtime = LazyHookRuntime([target])
    split_runtime.delta_backend = "triton-counter-inplace"
    split_runtime.field_policy = "target-split"
    split_runtime.qkv_kernel_policy = "split-launches"
    split_runtime.compute_dtype_policy = "float32"
    split_runtime.basis_by_site[target.site_id] = basis
    split_runtime.beta_by_target.update(beta_by_target)
    split_runtime.set_candidate_batch_by_order([cand_a, cand_b], prompt_count=1)
    split_runtime.update_row_candidates(["0", "1"], [0, 3, 7])

    packed_runtime = LazyHookRuntime([target])
    packed_runtime.delta_backend = "triton-counter-inplace"
    packed_runtime.field_policy = "target-split"
    packed_runtime.qkv_kernel_policy = "packed-qkv"
    packed_runtime.compute_dtype_policy = "float32"
    packed_runtime.basis_by_site[target.site_id] = basis
    packed_runtime.beta_by_target.update(beta_by_target)
    packed_runtime.set_candidate_batch_by_order([cand_a, cand_b], prompt_count=1)
    packed_runtime.update_row_candidates(["0", "1"], [0, 3, 7])

    split_base = base.clone()
    packed_base = base.clone()
    expected = split_runtime.delta(target, x, split_base)
    actual = packed_runtime.delta(target, x, packed_base)

    assert split_runtime._last_delta_is_output
    assert packed_runtime._last_delta_is_output
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    assert torch.equal(actual[:, 8:14], base[:, 8:14])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_vllm_lazy_hook_triton_counter_packed_qv_matches_fused_qkv_exact_offsets():
    target = HookTarget(
        module_name="model.layers.0.self_attn.qkv_proj",
        target_id="layer_0.self_attn.qkv_proj",
        site_id="layer_0.attn_in",
        layer_index=0,
        block_path="model.layers.0",
        suffix="qkv_proj",
        module=torch.nn.Linear(12, 20),
        input_dim=12,
        output_dim=20,
        fused_qkv_slices=("q_proj", "v_proj"),
        fused_q_out=8,
        fused_kv_out=6,
    )
    torch.manual_seed(6)
    basis = torch.randn((4, 12), dtype=torch.float32)
    x = torch.randn((5, 12), device="cuda", dtype=torch.float32)
    y = torch.zeros((5, 20), device="cuda", dtype=torch.float32)
    cand = replace(_candidate("a", 53), rng_version="counter_gaussian_v1", basis_rank=4, sign="+")
    beta_by_target = {
        "layer_0.self_attn.q_proj": 0.03,
        "layer_0.self_attn.v_proj": 0.07,
    }

    torch_runtime = LazyHookRuntime([target])
    torch_runtime.delta_backend = "torch"
    torch_runtime.field_policy = "fused-qkv-exact"
    torch_runtime.compute_dtype_policy = "float32"
    torch_runtime.basis_by_site[target.site_id] = basis
    torch_runtime.beta_by_target.update(beta_by_target)
    torch_runtime.set_candidate(cand)

    packed_runtime = LazyHookRuntime([target])
    packed_runtime.delta_backend = "triton-counter"
    packed_runtime.field_policy = "fused-qkv-exact"
    packed_runtime.qkv_kernel_policy = "packed-qkv"
    packed_runtime.compute_dtype_policy = "float32"
    packed_runtime.basis_by_site[target.site_id] = basis
    packed_runtime.beta_by_target.update(beta_by_target)
    packed_runtime.set_candidate(cand)

    expected = torch_runtime.delta(target, x, y)
    actual = packed_runtime.delta(target, x, y)

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    assert torch.count_nonzero(actual[:, 8:14]) == 0


class _FakeLazyRuntime:
    def __init__(self) -> None:
        self.active_candidate = None
        self.active_candidates = []
        self._order_prompt_count = 0
        self.qx_time_s = 0.0
        self.delta_time_s = 0.0
        self.stack_time_s = 0.0
        self.meta_time_s = 0.0
        self.kernel_time_s = 0.0
        self.delta_rows = 0
        self.delta_calls = 0

    def reset_timing(self) -> None:
        self.qx_time_s = 0.0
        self.delta_time_s = 0.0
        self.stack_time_s = 0.0
        self.meta_time_s = 0.0
        self.kernel_time_s = 0.0
        self.delta_rows = 0
        self.delta_calls = 0

    def set_candidate(self, candidate) -> None:
        self.active_candidate = candidate
        self.active_candidates = []
        self._order_prompt_count = 0

    def set_candidate_batch_by_order(self, candidates, *, prompt_count: int) -> None:
        self.active_candidate = None
        self.active_candidates = list(candidates)
        self._order_prompt_count = int(prompt_count)


class _FakeLLM:
    def __init__(self, runtime: _FakeLazyRuntime) -> None:
        self.runtime = runtime
        self.calls: list[list[str]] = []

    def generate(self, prompts, sampling, use_tqdm=False):
        del sampling, use_tqdm
        prompts = list(prompts)
        self.calls.append(prompts)
        self.runtime.delta_rows += len(prompts)
        self.runtime.delta_calls += 1
        outputs = []
        if self.runtime.active_candidate is not None:
            for prompt in prompts:
                text = f"{self.runtime.active_candidate.candidate_id}:{prompt}"
                outputs.append(SimpleNamespace(outputs=[SimpleNamespace(text=text, token_ids=[1])]))
            return outputs
        prompt_count = self.runtime._order_prompt_count
        for index, prompt in enumerate(prompts):
            candidate = self.runtime.active_candidates[index // prompt_count]
            text = f"{candidate.candidate_id}:{prompt}"
            outputs.append(SimpleNamespace(outputs=[SimpleNamespace(text=text, token_ids=[1])]))
        return outputs


def test_vllm_lazy_k1_prompt_microbatch_preserves_candidate_output_order(monkeypatch):
    cand_a = _candidate("a", 11)
    cand_b = _candidate("b", 29)
    runtime = _FakeLazyRuntime()
    llm = _FakeLLM(runtime)
    examples = [SimpleNamespace(id=index) for index in range(5)]
    prompt_inputs = [f"p{index}" for index in range(5)]

    def fake_score_outputs(examples_arg, outputs, *, max_new_tokens):
        del max_new_tokens
        rows = [
            {"example_id": example.id, "text": output.outputs[0].text, "output_tokens": 1, "exact": 1.0}
            for example, output in zip(examples_arg, outputs)
        ]
        return 1.0, len(rows), rows

    monkeypatch.setattr(lazy_k1, "_score_outputs", fake_score_outputs)

    score_rows, per_prompt_rows, timing = lazy_k1._evaluate_candidates(
        llm=llm,
        sampling=object(),
        runtime=runtime,
        examples=examples,
        prompt_inputs=prompt_inputs,
        candidates=[cand_a, cand_b],
        candidate_batch_size=2,
        prompt_batch_size=2,
        max_new_tokens=8,
        base_score=0.0,
        stage="test",
        source="source",
    )

    assert [len(call) for call in llm.calls] == [4, 4, 2]
    assert [row["candidate_id"] for row in score_rows] == ["a", "b"]
    assert [row["text"] for row in per_prompt_rows if row["candidate_id"] == "a"] == [
        "a:p0",
        "a:p1",
        "a:p2",
        "a:p3",
        "a:p4",
    ]
    assert [row["text"] for row in per_prompt_rows if row["candidate_id"] == "b"] == [
        "b:p0",
        "b:p1",
        "b:p2",
        "b:p3",
        "b:p4",
    ]
    assert timing["delta_rows"] == 10


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
    q_a = tensors["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"]
    v_a = tensors["base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight"]
    assert q_a.data_ptr() != v_a.data_ptr()


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


def test_vllm_lazy_replay_fused_qkv_matches_target_split_requested_slices():
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
        module_name="model.layers.0.self_attn.qkv_proj",
        target_id="layer_0.self_attn.qkv_proj",
        site_id="layer_0.attn_in",
        layer_index=0,
        block_path="model.layers.0",
        suffix="qkv_proj",
        module=torch.nn.Linear(4, 8),
        input_dim=4,
        output_dim=8,
        fused_qkv_slices=("q_proj", "v_proj"),
        fused_q_out=4,
        fused_kv_out=2,
    )
    runtime = LazyHookRuntime([target])
    runtime.basis_by_site[target.site_id] = basis[:2].contiguous()
    runtime.beta_by_target.update(_load_betas(source_summary, radius=0.4, scale_multiplier=2.0))
    runtime.set_candidate(candidate)
    x = torch.tensor([[1.0, -2.0, 0.5, 3.0], [0.25, 1.5, -1.0, 2.0]])
    y = torch.zeros((2, 8))

    delta = runtime.delta(target, x, y)
    tensors = subspace_lora_tensors(
        config=_TinyQwenConfig(),
        state_payload=state_payload,
        state_summary=state_summary,
        source_summary=source_summary,
        candidate=candidate,
        targets=["q_proj", "v_proj"],
        policy="target-split",
        tensor_dtype="float32",
        adapter_rank=2,
        scale_multiplier=2.0,
    )
    q_a = tensors["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"]
    q_b = tensors["base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"]
    v_a = tensors["base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight"]
    v_b = tensors["base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight"]
    expected = torch.zeros_like(y)
    expected[:, :4] = (x @ q_a.T) @ q_b.T
    expected[:, 6:8] = (x @ v_a.T) @ v_b.T

    assert torch.allclose(delta, expected)
    assert torch.equal(delta[:, 4:6], torch.zeros_like(delta[:, 4:6]))


def test_vllm_lazy_replay_fused_qkv_exact_field_policy_matches_adapter_slices():
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
        module_name="model.layers.0.self_attn.qkv_proj",
        target_id="layer_0.self_attn.qkv_proj",
        site_id="layer_0.attn_in",
        layer_index=0,
        block_path="model.layers.0",
        suffix="qkv_proj",
        module=torch.nn.Linear(4, 8),
        input_dim=4,
        output_dim=8,
        fused_qkv_slices=("q_proj", "v_proj"),
        fused_q_out=4,
        fused_kv_out=2,
    )
    runtime = LazyHookRuntime([target])
    runtime.field_policy = "fused-qkv-exact"
    runtime.basis_by_site[target.site_id] = basis[:2].contiguous()
    runtime.beta_by_target.update(_load_betas(source_summary, radius=0.4, scale_multiplier=2.0))
    runtime.set_candidate(candidate)
    x = torch.tensor([[1.0, -2.0, 0.5, 3.0], [0.25, 1.5, -1.0, 2.0]])
    y = torch.zeros((2, 8))

    delta = runtime.delta(target, x, y)
    tensors = subspace_lora_tensors(
        config=_TinyQwenConfig(),
        state_payload=state_payload,
        state_summary=state_summary,
        source_summary=source_summary,
        candidate=candidate,
        targets=["q_proj", "v_proj"],
        policy="fused-qkv-exact",
        tensor_dtype="float32",
        adapter_rank=2,
        scale_multiplier=2.0,
    )
    q_a = tensors["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"]
    q_b = tensors["base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"]
    v_a = tensors["base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight"]
    v_b = tensors["base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight"]
    expected = torch.zeros_like(y)
    expected[:, :4] = (x @ q_a.T) @ q_b.T
    expected[:, 6:8] = (x @ v_a.T) @ v_b.T

    assert torch.allclose(delta, expected)
    assert torch.equal(delta[:, 4:6], torch.zeros_like(delta[:, 4:6]))


def test_vllm_lazy_replay_bfloat16_scales_field_before_matmul_like_adapter():
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
    runtime.compute_dtype_policy = "bfloat16"
    runtime.basis_by_site[target.site_id] = basis[:2].contiguous()
    runtime.beta_by_target.update(_load_betas(source_summary, radius=0.4, scale_multiplier=2.0))
    runtime.set_candidate(candidate)
    x = torch.tensor([[1.0, -2.0, 0.5, 3.0], [0.25, 1.5, -1.0, 2.0]], dtype=torch.bfloat16)
    y = torch.zeros((2, 4), dtype=torch.bfloat16)

    delta = runtime.delta(target, x, y)
    tensors = subspace_lora_tensors(
        config=_TinyQwenConfig(),
        state_payload=state_payload,
        state_summary=state_summary,
        source_summary=source_summary,
        candidate=candidate,
        targets=["q_proj"],
        policy="target-split",
        tensor_dtype="bfloat16",
        adapter_rank=2,
        scale_multiplier=2.0,
    )
    a = tensors["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"]
    b = tensors["base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"]
    expected = (x @ a.T) @ b.T

    assert torch.equal(delta, expected)


def test_vllm_lazy_replay_honors_gaussian_hash_candidate_rng():
    target = HookTarget(
        module_name="model.layers.0.self_attn.q_proj",
        target_id="layer_0.self_attn.q_proj",
        site_id="layer_0.attn_in",
        layer_index=0,
        block_path="model.layers.0",
        suffix="q_proj",
        module=torch.nn.Linear(2, 3),
    )
    candidate = replace(_subspace_candidate_for_adapter(), rng_version="gaussian_hash_v1", basis_rank=2)
    runtime = LazyHookRuntime([target])
    field = runtime.field(target, candidate, output_dim=3, rank=2, device=torch.device("cpu"), dtype=torch.float32)

    expected = torch.tensor(
        [
            [
                gaussian_hash_v1(direction_seed=11, target_id=target.target_id, output_index=out_idx, basis_index=basis_idx)
                for basis_idx in range(2)
            ]
            for out_idx in range(3)
        ],
        dtype=torch.float32,
    )
    assert torch.equal(field, expected)


def test_counter_gaussian_v1_field_is_candidate_order_independent():
    target_hash = stable_u32("layer_0.self_attn.q_proj")
    first = counter_gaussian_v1(direction_seed=11, target_hash=target_hash, output_index=3, basis_index=1)
    second = counter_gaussian_v1(direction_seed=11, target_hash=target_hash, output_index=3, basis_index=1)
    opposite = counter_gaussian_v1(direction_seed=11, target_hash=target_hash, output_index=3, basis_index=1, sign="-")

    assert first == second
    assert opposite == -first


def test_vllm_lazy_random_field_is_device_independent_when_cuda_available():
    if not torch.cuda.is_available():
        return
    target = HookTarget(
        module_name="model.layers.0.self_attn.q_proj",
        target_id="layer_0.self_attn.q_proj",
        site_id="layer_0.attn_in",
        layer_index=0,
        block_path="model.layers.0",
        suffix="q_proj",
        module=torch.nn.Linear(4, 4),
    )
    candidate = replace(_subspace_candidate_for_adapter(), basis_rank=3)
    runtime = LazyHookRuntime([target])

    cpu_field = runtime.scaled_field(
        target,
        candidate,
        output_dim=4,
        rank=2,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        beta=0.25,
    )
    cuda_field = runtime.scaled_field(
        target,
        candidate,
        output_dim=4,
        rank=2,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
        beta=0.25,
    )

    assert torch.equal(cpu_field, cuda_field.cpu())


def test_vllm_lazy_replay_filter_maps_requested_qv_to_fused_qkv_target():
    fused = HookTarget(
        module_name="model.layers.0.self_attn.qkv_proj",
        target_id="layer_0.self_attn.qkv_proj",
        site_id="layer_0.attn_in",
        layer_index=0,
        block_path="model.layers.0",
        suffix="qkv_proj",
        module=torch.nn.Linear(4, 8),
    )

    targets = _filter_targets([fused], ["q_proj", "v_proj"], qkv_dims=(4, 2))

    assert len(targets) == 1
    assert targets[0].suffix == "qkv_proj"
    assert targets[0].fused_qkv_slices == ("q_proj", "v_proj")
    assert targets[0].fused_q_out == 4
    assert targets[0].fused_kv_out == 2
