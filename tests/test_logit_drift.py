import unittest

import torch

from randopt_lora_lab.logit_drift import drift_metrics, kl_from_logits, summarize


class LogitDriftTests(unittest.TestCase):
    def test_kl_is_zero_for_identical_logits(self):
        logits = torch.tensor([[1.0, 0.0, -1.0], [0.5, 0.25, -0.25]])

        kl = kl_from_logits(logits, logits)

        self.assertTrue(torch.allclose(kl, torch.zeros_like(kl), atol=1e-7))

    def test_kl_is_nonnegative_for_different_logits(self):
        left = torch.tensor([[2.0, 0.0, -1.0]])
        right = torch.tensor([[-1.0, 0.0, 2.0]])

        kl = kl_from_logits(left, right)

        self.assertGreaterEqual(float(kl.item()), 0.0)

    def test_drift_metrics_report_top1_agreement(self):
        base = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
        candidate = torch.tensor([[1.0, 0.0], [4.0, 3.0]])

        row = drift_metrics(base, candidate)

        self.assertEqual(row["prompts"], 2)
        self.assertEqual(row["top1_equal_rate"], 0.5)
        self.assertGreater(row["logit_l2_mean"], 0.0)
        self.assertGreater(row["kl_base_to_candidate_mean"], 0.0)

    def test_summary_requires_configured_gates(self):
        rows = [{"kl_base_to_candidate_mean": 0.01, "kl_candidate_to_base_mean": 0.02, "logit_l2_mean": 1.0, "top1_equal_rate": 1.0}]

        passed = summarize(rows, max_mean_kl=0.02, min_top1_equal=1.0)
        failed = summarize(rows, max_mean_kl=0.005, min_top1_equal=1.0)

        self.assertTrue(passed["pass"])
        self.assertFalse(failed["pass"])


if __name__ == "__main__":
    unittest.main()
