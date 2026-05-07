import tempfile
import types
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file

from randopt_lora_lab.dense_space import dense_noise_tensor
from randopt_lora_lab.gaussian_parity import best_rank_projection, lora_update
from randopt_lora_lab.lora_space import Candidate, canonical_module_name, lora_noise_tensors
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


if __name__ == "__main__":
    unittest.main()
