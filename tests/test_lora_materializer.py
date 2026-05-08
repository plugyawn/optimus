import tempfile
import types
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file

from randopt_lora_lab.dense_space import dense_noise_tensor
from randopt_lora_lab.gaussian_parity import best_rank_projection, lora_update
from randopt_lora_lab.lora_space import (
    Candidate,
    activation_spectral_scale,
    activation_spectral_uses_singular_values,
    canonical_module_name,
    lora_noise_tensors,
    sparse_lora_density,
    spectral_projected_scale,
)
from randopt_lora_lab.vllm_lora_bench import save_seed_adapter


class LoraMaterializerTests(unittest.TestCase):
    def test_canonical_module_name_strips_peft_prefix(self):
        peft_name = "base_model.model.model.layers.0.self_attn.q_proj"
        self.assertEqual(canonical_module_name(peft_name), "model.layers.0.self_attn.q_proj")
        self.assertEqual(canonical_module_name("model.layers.1.self_attn.v_proj"), "model.layers.1.self_attn.v_proj")

    def test_peft_and_vllm_materialization_share_tensors(self):
        candidate = Candidate("isotropic", seed=123, sigma=0.0075, sign=-1)
        rank = 4
        hidden = 16
        config = types.SimpleNamespace(
            hidden_size=hidden,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
        )
        module = "model.layers.0.self_attn.q_proj"
        expected_a, expected_b = lora_noise_tensors(module, (rank, hidden), (hidden, rank), candidate, rank)
        with tempfile.TemporaryDirectory() as tmp:
            save_seed_adapter(
                Path(tmp),
                model="Qwen/Qwen2.5-3B-Instruct",
                candidate=candidate,
                rank=rank,
                targets=["q_proj"],
                config=config,
                tensor_dtype="float32",
            )
            tensors = load_file(str(Path(tmp) / "adapter_model.safetensors"))
        prefix = f"base_model.model.{module}"
        self.assertTrue(torch.equal(tensors[f"{prefix}.lora_A.weight"], expected_a))
        self.assertTrue(torch.equal(tensors[f"{prefix}.lora_B.weight"], expected_b))

    def test_projected_gaussian_family_factors_dense_projection(self):
        candidate = Candidate("projected_gaussian_rank_r", seed=456, sigma=0.01, sign=1)
        rank = 3
        module = "model.layers.0.self_attn.q_proj"

        a, b = lora_noise_tensors(module, (rank, 8), (7, rank), candidate, rank)
        dense = dense_noise_tensor(module, (7, 8), Candidate("dense_gaussian", seed=456, sigma=0.01, sign=1))
        projected = best_rank_projection(dense, rank)

        self.assertEqual(tuple(a.shape), (rank, 8))
        self.assertEqual(tuple(b.shape), (7, rank))
        self.assertTrue(torch.allclose(lora_update(a, b), projected.to(a.dtype), atol=1e-6, rtol=1e-6))

    def test_randomized_projected_gaussian_family_is_deterministic_low_rank(self):
        candidate = Candidate("randomized_projected_gaussian_rank_r", seed=456, sigma=0.01, sign=1)
        rank = 3
        module = "model.layers.0.self_attn.q_proj"

        a1, b1 = lora_noise_tensors(module, (rank, 8), (7, rank), candidate, rank)
        a2, b2 = lora_noise_tensors(module, (rank, 8), (7, rank), candidate, rank)
        update = lora_update(a1.double(), b1.double())

        self.assertEqual(tuple(a1.shape), (rank, 8))
        self.assertEqual(tuple(b1.shape), (7, rank))
        self.assertTrue(torch.equal(a1, a2))
        self.assertTrue(torch.equal(b1, b2))
        self.assertLessEqual(int(torch.linalg.matrix_rank(update, atol=1e-6).item()), rank)
        self.assertTrue(torch.isfinite(update).all())

    def test_spectral_projected_gaussian_family_is_deterministic_low_rank(self):
        candidate = Candidate("spectral_projected_gaussian_rank_r", seed=456, sigma=0.01, sign=-1)
        rank = 3
        module = "model.layers.0.self_attn.q_proj"

        a1, b1 = lora_noise_tensors(module, (rank, 8), (7, rank), candidate, rank)
        a2, b2 = lora_noise_tensors(module, (rank, 8), (7, rank), candidate, rank)
        update = lora_update(a1.double(), b1.double())

        self.assertEqual(tuple(a1.shape), (rank, 8))
        self.assertEqual(tuple(b1.shape), (7, rank))
        self.assertTrue(torch.equal(a1, a2))
        self.assertTrue(torch.equal(b1, b2))
        self.assertLessEqual(int(torch.linalg.matrix_rank(update, atol=1e-6).item()), rank)
        self.assertTrue(torch.isfinite(update).all())

    def test_spectral_projected_scale_variants_change_update_norm(self):
        rank = 3
        module = "model.layers.0.self_attn.q_proj"
        base = Candidate("spectral_projected_gaussian_rank_r", seed=456, sigma=0.01, sign=1)
        scaled = Candidate("spectral_projected_gaussian_rank_r_c2", seed=456, sigma=0.01, sign=1)

        a_base, b_base = lora_noise_tensors(module, (rank, 8), (7, rank), base, rank)
        a_scaled, b_scaled = lora_noise_tensors(module, (rank, 8), (7, rank), scaled, rank)

        self.assertEqual(spectral_projected_scale(scaled.family), 2.0)
        self.assertGreater(torch.linalg.norm(lora_update(a_scaled, b_scaled)), torch.linalg.norm(lora_update(a_base, b_base)))

    def test_activation_spectral_lora_uses_activation_basis(self):
        rank = 3
        module = "base_model.model.model.layers.0.self_attn.q_proj"
        candidate = Candidate("activation_spectral_lora_c2", seed=456, sigma=0.01, sign=1)
        basis = torch.eye(8)[:rank].contiguous()

        a1, b1 = lora_noise_tensors(
            module,
            (rank, 8),
            (7, rank),
            candidate,
            rank,
            family_state={module: basis},
            state_key=module,
        )
        a2, b2 = lora_noise_tensors(
            module,
            (rank, 8),
            (7, rank),
            candidate,
            rank,
            family_state={module: basis},
            state_key=module,
        )
        update = lora_update(a1.double(), b1.double())

        self.assertEqual(activation_spectral_scale(candidate.family), 2.0)
        self.assertEqual(tuple(a1.shape), (rank, 8))
        self.assertEqual(tuple(b1.shape), (7, rank))
        self.assertTrue(torch.equal(a1, a2))
        self.assertTrue(torch.equal(b1, b2))
        self.assertLessEqual(int(torch.linalg.matrix_rank(update, atol=1e-6).item()), rank)
        self.assertTrue(torch.isfinite(update).all())
        self.assertTrue(torch.allclose(a1[:, rank:], torch.zeros_like(a1[:, rank:])))

    def test_activation_spectral_lora_requires_state(self):
        candidate = Candidate("activation_spectral_lora", seed=456, sigma=0.01, sign=1)
        with self.assertRaisesRegex(ValueError, "requires an activation basis"):
            lora_noise_tensors("model.layers.0.self_attn.q_proj", (3, 8), (7, 3), candidate, 3)

    def test_activation_spectral_adapter_materialization_uses_peft_state_key(self):
        candidate = Candidate("activation_spectral_lora", seed=456, sigma=0.01, sign=1)
        rank = 3
        hidden = 8
        module = "model.layers.0.self_attn.q_proj"
        peft_module = f"base_model.model.{module}"
        family_state = {peft_module: torch.eye(hidden)[:rank].contiguous()}
        config = types.SimpleNamespace(
            hidden_size=hidden,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
        expected_a, expected_b = lora_noise_tensors(
            module,
            (rank, hidden),
            (hidden, rank),
            candidate,
            rank,
            family_state=family_state,
            state_key=module,
        )
        with tempfile.TemporaryDirectory() as tmp:
            save_seed_adapter(
                Path(tmp),
                model="Qwen/Qwen2.5-3B-Instruct",
                candidate=candidate,
                rank=rank,
                targets=["q_proj"],
                config=config,
                tensor_dtype="float32",
                family_state=family_state,
            )
            tensors = load_file(str(Path(tmp) / "adapter_model.safetensors"))
        prefix = f"base_model.model.{module}"
        self.assertTrue(torch.equal(tensors[f"{prefix}.lora_A.weight"], expected_a))
        self.assertTrue(torch.equal(tensors[f"{prefix}.lora_B.weight"], expected_b))

    def test_activation_spectral_sv_uses_singular_value_weights(self):
        rank = 3
        module = "model.layers.0.self_attn.q_proj"
        basis = torch.eye(8)[:rank].contiguous()
        family_state = {
            module: {
                "basis": basis,
                "singular_values": torch.tensor([9.0, 4.0, 1.0]),
            }
        }
        flat = Candidate("activation_spectral_lora", seed=456, sigma=0.01, sign=1)
        weighted = Candidate("activation_spectral_lora_sv", seed=456, sigma=0.01, sign=1)

        flat_a, _ = lora_noise_tensors(module, (rank, 8), (7, rank), flat, rank, family_state=family_state)
        weighted_a, _ = lora_noise_tensors(module, (rank, 8), (7, rank), weighted, rank, family_state=family_state)

        self.assertFalse(activation_spectral_uses_singular_values(flat.family))
        self.assertTrue(activation_spectral_uses_singular_values(weighted.family))
        self.assertTrue(torch.allclose(flat_a.norm(dim=1), flat_a.norm(dim=1).mean().expand(rank)))
        self.assertGreater(float(weighted_a.norm(dim=1).max() - weighted_a.norm(dim=1).min()), 0.0)

    def test_sparse_low_rank_lora_materializes_sparse_scaled_factors(self):
        candidate = Candidate("sparse_low_rank_lora_d0p25", seed=789, sigma=0.01, sign=1)
        rank = 8
        module = "model.layers.0.self_attn.q_proj"

        a, b = lora_noise_tensors(module, (rank, 512), (512, rank), candidate, rank)

        self.assertEqual(sparse_lora_density(candidate.family), 0.25)
        self.assertGreater(float((a == 0).float().mean()), 0.65)
        self.assertGreater(float((b == 0).float().mean()), 0.65)
        self.assertTrue(torch.isfinite(lora_update(a, b)).all())


if __name__ == "__main__":
    unittest.main()
