import math
import tempfile
import types
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file

from optimus.core.perturbations import PerturbationSpec as Candidate
from optimus.core.perturbations import canonical_module_name
from optimus.modeling import qwen_lora_shapes, save_seed_adapter
from optimus.modeling.qwen import validate_qwen_lora_config
from optimus.modeling.dense import dense_noise_tensor
from optimus.modeling.geometry import best_rank_projection, lora_update
from optimus.modeling.noise import (
    activation_basis_spectral_scale,
    activation_generalized_projected_scale,
    activation_generalized_spectral_scale,
    activation_projected_scale,
    activation_spectral_scale,
    activation_spectral_uses_singular_values,
    generalized_activation_basis,
    lora_noise_tensors,
    sparse_lora_density,
    spectral_projected_scale,
)


class LoraMaterializerTests(unittest.TestCase):
    def test_canonical_module_name_strips_peft_prefixes(self):
        self.assertEqual(
            canonical_module_name("base_model.model.model.layers.0.self_attn.q_proj"),
            "model.layers.0.self_attn.q_proj",
        )
        self.assertEqual(
            canonical_module_name("base_model.model.model.language_model.layers.0.self_attn.q_proj"),
            "model.language_model.layers.0.self_attn.q_proj",
        )

    def test_peft_adapter_materialization_uses_deterministic_lora_tensors(self):
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
                model="Qwen/Qwen3-4B",
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

    def test_qwen3_text_config_is_validated_for_direct_lora(self):
        config = types.SimpleNamespace(
            model_type="qwen3",
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
        )

        validate_qwen_lora_config(config, model_name="Qwen/Qwen3-4B")

        self.assertIn(("model.layers.0.self_attn.v_proj", 16, 8), qwen_lora_shapes(config, ["v_proj"]))
        self.assertIn(("model.layers.0.self_attn.q_proj", 16, 16), qwen_lora_shapes(config, ["q_proj"]))
        self.assertIn(("model.layers.0.self_attn.o_proj", 16, 16), qwen_lora_shapes(config, ["o_proj"]))

    def test_qwen3_projection_shapes_follow_explicit_head_dim(self):
        config = types.SimpleNamespace(
            model_type="qwen3",
            hidden_size=2560,
            intermediate_size=9728,
            num_hidden_layers=1,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
        )

        self.assertEqual(
            qwen_lora_shapes(config, ["q_proj", "k_proj", "v_proj", "o_proj"]),
            [
                ("model.layers.0.self_attn.q_proj", 2560, 4096),
                ("model.layers.0.self_attn.k_proj", 2560, 1024),
                ("model.layers.0.self_attn.v_proj", 2560, 1024),
                ("model.layers.0.self_attn.o_proj", 4096, 2560),
            ],
        )

    def test_qwen3_vl_materialization_uses_language_model_prefix(self):
        candidate = Candidate("isotropic", seed=123, sigma=0.0075, sign=-1)
        rank = 4
        hidden = 16
        config = types.SimpleNamespace(
            model_type="qwen3_vl_text",
            hidden_size=hidden,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
        )
        module = "model.language_model.layers.0.self_attn.k_proj"
        self.assertIn((module, hidden, 8), qwen_lora_shapes(config, ["k_proj"]))
        expected_a, expected_b = lora_noise_tensors(module, (rank, hidden), (8, rank), candidate, rank)

        with tempfile.TemporaryDirectory() as tmp:
            save_seed_adapter(
                Path(tmp),
                model="Qwen/Qwen3-VL-8B-Instruct",
                candidate=candidate,
                rank=rank,
                targets=["k_proj"],
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
        dense = dense_noise_tensor(module, (7, 8), Candidate("dense_gaussian", seed=456, sigma=0.01, sign=1, method="dense"))
        projected = best_rank_projection(dense, rank)

        self.assertEqual(tuple(a.shape), (rank, 8))
        self.assertEqual(tuple(b.shape), (7, rank))
        self.assertTrue(torch.allclose(lora_update(a, b), projected.to(a.dtype), atol=1e-6, rtol=1e-6))

    def test_spectral_projected_scale_variants_change_update_norm(self):
        rank = 3
        module = "model.layers.0.self_attn.q_proj"
        base = Candidate("spectral_projected_gaussian_rank_r", seed=456, sigma=0.01, sign=1)
        scaled = Candidate("spectral_projected_gaussian_rank_r_c2", seed=456, sigma=0.01, sign=1)

        a_base, b_base = lora_noise_tensors(module, (rank, 8), (7, rank), base, rank)
        a_scaled, b_scaled = lora_noise_tensors(module, (rank, 8), (7, rank), scaled, rank)

        self.assertEqual(spectral_projected_scale(scaled.family), 2.0)
        self.assertGreater(torch.linalg.norm(lora_update(a_scaled, b_scaled)), torch.linalg.norm(lora_update(a_base, b_base)))

    def test_activation_projected_gaussian_preserves_dense_seed_in_basis(self):
        rank = 3
        module = "base_model.model.model.layers.0.self_attn.q_proj"
        candidate = Candidate("activation_projected_gaussian_rank_r_c2", seed=456, sigma=0.01, sign=-1)
        basis = torch.eye(8)[:rank].contiguous()

        a, b = lora_noise_tensors(
            module,
            (rank, 8),
            (7, rank),
            candidate,
            rank,
            family_state={module: basis},
            state_key=module,
        )
        dense = dense_noise_tensor(
            "model.layers.0.self_attn.q_proj",
            (7, 8),
            Candidate("dense_gaussian", seed=456, sigma=0.02, sign=-1, method="dense"),
        )
        expected = torch.zeros_like(dense)
        expected[:, :rank] = dense[:, :rank]

        self.assertEqual(activation_projected_scale(candidate.family), 2.0)
        self.assertTrue(torch.allclose(lora_update(a, b), expected, atol=1e-6, rtol=1e-6))

    def test_activation_generalized_projected_gaussian_uses_same_projection_path(self):
        rank = 3
        module = "base_model.model.model.layers.0.self_attn.q_proj"
        candidate = Candidate("activation_generalized_projected_gaussian_rank_r_c2", seed=456, sigma=0.01, sign=-1)
        basis = torch.eye(8)[:rank].contiguous()

        a, b = lora_noise_tensors(
            module,
            (rank, 8),
            (7, rank),
            candidate,
            rank,
            family_state={module: {"basis": basis}},
            state_key=module,
        )
        dense = dense_noise_tensor(
            "model.layers.0.self_attn.q_proj",
            (7, 8),
            Candidate("dense_gaussian", seed=456, sigma=0.02, sign=-1, method="dense"),
        )
        expected = torch.zeros_like(dense)
        expected[:, :rank] = dense[:, :rank]

        self.assertEqual(activation_generalized_projected_scale(candidate.family), 2.0)
        self.assertTrue(torch.allclose(lora_update(a, b), expected, atol=1e-6, rtol=1e-6))

    def test_generalized_activation_basis_prefers_target_over_anchor_energy(self):
        target = torch.tensor(
            [
                [5.0, 0.1, 0.0, 0.0],
                [4.0, -0.1, 0.0, 0.0],
                [-5.0, 0.0, 0.0, 0.0],
                [-4.0, 0.0, 0.0, 0.0],
            ]
        )
        anchor = torch.tensor(
            [
                [0.0, 5.0, 0.0, 0.0],
                [0.0, 4.0, 0.0, 0.0],
                [0.0, -5.0, 0.0, 0.0],
                [0.0, -4.0, 0.0, 0.0],
            ]
        )

        basis, scores = generalized_activation_basis(target, anchor, rank=2)

        self.assertEqual(tuple(basis.shape), (2, 4))
        self.assertEqual(tuple(scores.shape), (2,))
        self.assertGreater(abs(float(basis[0, 0])), 0.95)
        self.assertGreaterEqual(float(scores[0]), float(scores[1]))

    def test_activation_spectral_lora_requires_state_and_uses_singular_weights(self):
        missing = Candidate("activation_spectral_lora", seed=456, sigma=0.01, sign=1)
        with self.assertRaisesRegex(ValueError, "requires an activation basis"):
            lora_noise_tensors("model.layers.0.self_attn.q_proj", (3, 8), (7, 3), missing, 3)

        rank = 3
        module = "model.layers.0.self_attn.q_proj"
        family_state = {
            module: {
                "basis": torch.eye(8)[:rank].contiguous(),
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

    def test_activation_target_scale_family_uses_module_specific_scale(self):
        rank = 3
        family = "activation_spectral_lora_tscale_q2_v1p5"
        candidate = Candidate(family, seed=456, sigma=0.01, sign=1)
        family_state = {
            "model.layers.0.self_attn.q_proj": torch.eye(8)[:rank].contiguous(),
            "model.layers.0.self_attn.v_proj": torch.eye(8)[:rank].contiguous(),
        }

        q_a, q_b = lora_noise_tensors(
            "model.layers.0.self_attn.q_proj",
            (rank, 8),
            (8, rank),
            candidate,
            rank,
            family_state=family_state,
        )
        v_a, v_b = lora_noise_tensors(
            "model.layers.0.self_attn.v_proj",
            (rank, 8),
            (4, rank),
            candidate,
            rank,
            family_state=family_state,
        )

        self.assertEqual(activation_basis_spectral_scale(family, "model.layers.0.self_attn.q_proj"), 2.0)
        self.assertEqual(activation_basis_spectral_scale(family, "model.layers.0.self_attn.v_proj"), 1.5)
        q_expected = 0.01 * 2.0 * (math.sqrt(8.0) + math.sqrt(8.0)) * math.sqrt(rank)
        v_expected = 0.01 * 1.5 * (math.sqrt(4.0) + math.sqrt(8.0)) * math.sqrt(rank)
        self.assertTrue(
            torch.allclose(lora_update(q_a.double(), q_b.double()).norm(), torch.tensor(q_expected, dtype=torch.float64), rtol=1e-5)
        )
        self.assertTrue(
            torch.allclose(lora_update(v_a.double(), v_b.double()).norm(), torch.tensor(v_expected, dtype=torch.float64), rtol=1e-5)
        )

    def test_activation_generalized_spectral_lora_and_sparse_lora(self):
        rank = 3
        module = "model.layers.0.self_attn.q_proj"
        family_state = {module: {"basis": torch.eye(8)[:rank].contiguous(), "singular_values": torch.tensor([3.0, 2.0, 1.0])}}
        candidate = Candidate("activation_generalized_spectral_lora_c2", seed=456, sigma=0.01, sign=1)

        a, b = lora_noise_tensors(module, (rank, 8), (7, rank), candidate, rank, family_state=family_state)
        update = lora_update(a.double(), b.double())

        self.assertEqual(activation_generalized_spectral_scale(candidate.family), 2.0)
        self.assertEqual(activation_spectral_scale("activation_spectral_lora_c2"), 2.0)
        self.assertLessEqual(int(torch.linalg.matrix_rank(update, atol=1e-6).item()), rank)

        sparse = Candidate("sparse_low_rank_lora_d0p25", seed=789, sigma=0.01, sign=1)
        sparse_a, sparse_b = lora_noise_tensors(module, (8, 512), (512, 8), sparse, 8)
        self.assertEqual(sparse_lora_density(sparse.family), 0.25)
        self.assertGreater(float((sparse_a == 0).float().mean()), 0.65)
        self.assertGreater(float((sparse_b == 0).float().mean()), 0.65)


if __name__ == "__main__":
    unittest.main()
