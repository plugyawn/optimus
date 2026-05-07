import unittest

from randopt_lora_lab.backend_rollout_ablation_report import summarize


class BackendRolloutAblationReportTests(unittest.TestCase):
    def test_summarize_requires_rollout_not_exact_parity(self):
        rows = [
            {
                "run": "default",
                "condition": "candidate",
                "text_equal_rate": 0.0,
                "exact_equal_rate": 1.0,
                "abs_cap_hit_delta": 1.0,
                "abs_malformed_delta": 0.0,
                "mean_abs_output_token_delta": 16.0,
            },
            {
                "run": "strict",
                "condition": "candidate",
                "text_equal_rate": 1.0,
                "exact_equal_rate": 1.0,
                "abs_cap_hit_delta": 0.0,
                "abs_malformed_delta": 0.0,
                "mean_abs_output_token_delta": 0.0,
            },
        ]

        summary = summarize(
            rows,
            min_text_equal=0.95,
            max_cap_delta=0.05,
            max_malformed_delta=0.05,
            max_token_delta=1.0,
        )

        self.assertFalse(summary["by_run"]["default"]["pass"])
        self.assertTrue(summary["by_run"]["strict"]["pass"])
        self.assertEqual(summary["passing_runs"], ["strict"])
        self.assertTrue(summary["pass"])


if __name__ == "__main__":
    unittest.main()
