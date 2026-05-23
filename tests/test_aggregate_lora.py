import unittest

import torch

from optimus.search.aggregation import aggregate_lora_tensors, normalized_weights
from optimus.modeling.geometry import lora_update
from randopt_lora_lab.lora_space import Candidate, lora_noise_tensors


class AggregateLoraTests(unittest.TestCase):
    def test_normalized_weights_have_unit_l2_norm(self):
        weights = normalized_weights([0.1, 0.3, 0.5], "score")

        self.assertAlmostEqual(sum(weight * weight for weight in weights), 1.0)
        self.assertEqual(weights[0], 0.0)

    def test_aggregate_concat_represents_weighted_sum(self):
        module = "model.layers.0.self_attn.q_proj"
        base_rank = 2
        candidates = [
            Candidate("factor_gaussian_lora", seed=11, sigma=0.01, sign=1),
            Candidate("factor_gaussian_lora", seed=22, sigma=0.01, sign=-1),
        ]
        weights = [0.6, -0.8]

        a_cat, b_cat = aggregate_lora_tensors(module, 5, 4, candidates, weights, base_rank)

        expected = torch.zeros((4, 5))
        for candidate, weight in zip(candidates, weights):
            a, b = lora_noise_tensors(module, (base_rank, 5), (4, base_rank), candidate, base_rank)
            expected += weight * lora_update(a, b)
        self.assertEqual(tuple(a_cat.shape), (4, 5))
        self.assertEqual(tuple(b_cat.shape), (4, 4))
        self.assertTrue(torch.allclose(lora_update(a_cat, b_cat), expected, atol=1e-7, rtol=1e-7))


if __name__ == "__main__":
    unittest.main()
