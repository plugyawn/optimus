import unittest

from randopt_lora_lab.backend_rollout_probe import summarize


class BackendRolloutProbeTests(unittest.TestCase):
    def test_summarize_reports_rollout_equality_rates(self):
        rows = [
            {
                "condition": "base",
                "text_equal": True,
                "answer_equal": True,
                "exact_equal": True,
                "hf_exact": 1.0,
                "vllm_exact": 1.0,
                "hf_malformed": 0.0,
                "vllm_malformed": 0.0,
                "hf_cap_hit": 0.0,
                "vllm_cap_hit": 0.0,
                "output_token_delta": 0,
            },
            {
                "condition": "base",
                "text_equal": False,
                "answer_equal": True,
                "exact_equal": True,
                "hf_exact": 0.0,
                "vllm_exact": 0.0,
                "hf_malformed": 0.0,
                "vllm_malformed": 1.0,
                "hf_cap_hit": 0.0,
                "vllm_cap_hit": 1.0,
                "output_token_delta": 3,
            },
        ]

        summary = summarize(rows)

        self.assertAlmostEqual(summary["overall_text_equal_rate"], 0.5)
        self.assertAlmostEqual(summary["overall_answer_equal_rate"], 1.0)
        self.assertAlmostEqual(summary["conditions"]["base"]["exact_equal_rate"], 1.0)
        self.assertAlmostEqual(summary["conditions"]["base"]["mean_abs_output_token_delta"], 1.5)


if __name__ == "__main__":
    unittest.main()
