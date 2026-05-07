import unittest

import torch

from randopt_lora_lab.gaussian_parity import (
    best_rank_projection,
    dense_gaussian_matrix,
    expected_update_stats,
    lora_update,
    low_rank_factors_from_dense,
    projection_stats,
    qwen25_3b_qv_specs,
    randomized_low_rank_factors_from_dense,
    required_rank_for_energy,
    summarize_specs,
)


class GaussianParityTests(unittest.TestCase):
    def test_lora_update_rank_is_capped(self):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(123)
        rank = 3
        a = torch.randn((rank, 8), generator=gen, dtype=torch.float64)
        b = torch.randn((7, rank), generator=gen, dtype=torch.float64)

        update = lora_update(a, b)

        self.assertLessEqual(int(torch.linalg.matrix_rank(update).item()), rank)

    def test_dense_gaussian_is_full_rank_for_test_shape(self):
        dense = dense_gaussian_matrix((7, 8), seed=456)

        self.assertEqual(int(torch.linalg.matrix_rank(dense).item()), 7)

    def test_best_rank_projection_has_residual_until_full_rank(self):
        dense = dense_gaussian_matrix((10, 8), seed=789)

        low = projection_stats(dense, [3])[0]
        full = projection_stats(dense, [8])[0]

        self.assertLess(low["captured_frob_fraction"], 1.0)
        self.assertGreater(low["relative_frob_error"], 0.0)
        self.assertAlmostEqual(full["captured_frob_fraction"], 1.0, places=12)
        self.assertAlmostEqual(full["relative_frob_error"], 0.0, places=12)

    def test_svd_projection_can_be_represented_as_lora_factors(self):
        dense = dense_gaussian_matrix((9, 7), seed=2468)
        rank = 4

        projected = best_rank_projection(dense, rank)
        a, b = low_rank_factors_from_dense(dense, rank)

        self.assertEqual(tuple(a.shape), (rank, 7))
        self.assertEqual(tuple(b.shape), (9, rank))
        self.assertTrue(torch.allclose(lora_update(a, b), projected, atol=1e-10, rtol=1e-10))

    def test_randomized_projection_is_low_rank_and_deterministic(self):
        dense = dense_gaussian_matrix((16, 12), seed=2469).float()
        rank = 4

        a1, b1 = randomized_low_rank_factors_from_dense(dense, rank, oversample=4, n_iter=1, seed=99)
        a2, b2 = randomized_low_rank_factors_from_dense(dense, rank, oversample=4, n_iter=1, seed=99)
        update = lora_update(a1, b1)

        self.assertEqual(tuple(a1.shape), (rank, 12))
        self.assertEqual(tuple(b1.shape), (16, rank))
        self.assertTrue(torch.equal(a1, a2))
        self.assertTrue(torch.equal(b1, b2))
        self.assertLessEqual(int(torch.linalg.matrix_rank(update.double(), atol=1e-6).item()), rank)
        self.assertLessEqual(float((update * update).sum().item()), float((dense * dense).sum().item()))
        self.assertGreater(float((update * update).sum().item()), 0.0)

    def test_required_rank_for_energy_is_monotonic(self):
        dense = dense_gaussian_matrix((16, 12), seed=1357)

        required = required_rank_for_energy(dense, [0.5, 0.9, 0.99])

        self.assertLessEqual(required[0.5], required[0.9])
        self.assertLessEqual(required[0.9], required[0.99])
        self.assertLessEqual(required[0.99], 12)

    def test_qwen_qv_rank8_capacity_is_tiny(self):
        summary = summarize_specs(qwen25_3b_qv_specs(rank=8))

        self.assertEqual(summary["total_dense_params"], 36 * (2048 * 2048 + 256 * 2048))
        self.assertEqual(summary["total_lora_params"], 36 * (8 * (2048 + 2048) + 8 * (256 + 2048)))
        self.assertLess(summary["total_param_fraction"], 0.011)
        self.assertLess(summary["summed_rank_fraction"], 0.02)

    def test_factor_lora_expected_frobenius_matches_dense_gaussian(self):
        specs = qwen25_3b_qv_specs(rank=8, layers=1)

        stats = expected_update_stats(specs, sigma=0.01)

        self.assertEqual(stats["total_expected_frob_ratio_factor_lora_over_dense"], 1.0)
        for row in stats["per_matrix"]:
            self.assertEqual(row["expected_frob_ratio_factor_lora_over_dense"], 1.0)
            self.assertLess(row["factor_lora_rank_cap"], row["dense_rank_almost_sure"])


if __name__ == "__main__":
    unittest.main()
