import unittest

import torch

from randopt_lora_lab.gaussian_parity import MatrixSpec
from randopt_lora_lab.lora_space import Candidate
from randopt_lora_lab.update_geometry import effective_rank, family_geometry, matrix_update_for_family


class UpdateGeometryTests(unittest.TestCase):
    def test_dense_gaussian_is_full_rank_for_test_shape(self):
        spec = MatrixSpec("model.layers.0.self_attn.q_proj", 12, 10, rank=4)
        candidate = Candidate("geometry", seed=123, sigma=0.01)

        delta = matrix_update_for_family(spec, candidate, "dense_gaussian")

        self.assertEqual(delta.shape, (12, 10))
        self.assertEqual(torch.linalg.matrix_rank(delta).item(), 10)

    def test_factor_lora_rank_is_capped(self):
        spec = MatrixSpec("model.layers.0.self_attn.q_proj", 12, 10, rank=4)
        candidate = Candidate("geometry", seed=123, sigma=0.01)

        delta = matrix_update_for_family(spec, candidate, "factor_gaussian_lora")

        self.assertLessEqual(torch.linalg.matrix_rank(delta).item(), 4)

    def test_projected_gaussian_rank_is_capped(self):
        spec = MatrixSpec("model.layers.0.self_attn.q_proj", 12, 10, rank=4)
        candidate = Candidate("geometry", seed=123, sigma=0.01)

        delta = matrix_update_for_family(spec, candidate, "projected_gaussian_rank_r")

        self.assertLessEqual(effective_rank(delta), 4)

    def test_family_geometry_separates_rank_fractions(self):
        specs = [
            MatrixSpec("model.layers.0.self_attn.q_proj", 12, 10, rank=4),
            MatrixSpec("model.layers.0.self_attn.v_proj", 8, 10, rank=4),
        ]
        candidate = Candidate("geometry", seed=456, sigma=0.01)

        result = family_geometry(
            specs,
            candidate,
            [
                "dense_gaussian",
                "factor_gaussian_lora",
                "projected_gaussian_rank_r",
                "randomized_projected_gaussian_rank_r",
                "spectral_projected_gaussian_rank_r",
                "spectral_projected_gaussian_rank_r_c0p5",
            ],
            sparsity_threshold=0.0,
        )

        dense = result["dense_gaussian"]["summary"]
        factor = result["factor_gaussian_lora"]["summary"]
        projected = result["projected_gaussian_rank_r"]["summary"]
        randomized = result["randomized_projected_gaussian_rank_r"]["summary"]
        spectral = result["spectral_projected_gaussian_rank_r"]["summary"]
        spectral_half = result["spectral_projected_gaussian_rank_r_c0p5"]["summary"]
        self.assertEqual(dense["total_l0_sparsity"], 0.0)
        self.assertGreater(dense["weighted_effective_rank_fraction"], factor["weighted_effective_rank_fraction"])
        self.assertEqual(factor["mean_effective_rank"], 4)
        self.assertEqual(projected["mean_effective_rank"], 4)
        self.assertEqual(randomized["mean_effective_rank"], 4)
        self.assertEqual(spectral["mean_effective_rank"], 4)
        self.assertEqual(spectral_half["mean_effective_rank"], 4)


if __name__ == "__main__":
    unittest.main()
