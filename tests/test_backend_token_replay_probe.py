import unittest

from randopt_lora_lab.backend_token_replay_probe import contains_suffix, next_prefix_id, summarize


class BackendTokenReplayProbeTests(unittest.TestCase):
    def test_contains_suffix(self):
        self.assertTrue(contains_suffix([1, 2, 3, 4], [3, 4]))
        self.assertFalse(contains_suffix([1, 2, 3], [3, 4]))
        self.assertFalse(contains_suffix([1, 2, 3], []))

    def test_next_prefix_id_modes(self):
        self.assertEqual(next_prefix_id("hf", 5, 6, False), 5)
        self.assertEqual(next_prefix_id("vllm", 5, 6, False), 6)
        self.assertEqual(next_prefix_id("match", 5, 5, True), 5)
        self.assertIsNone(next_prefix_id("match", 5, 6, False))
        with self.assertRaises(ValueError):
            next_prefix_id("bad", 5, 6, False)

    def test_summarize_groups_by_condition_and_prefix_mode(self):
        rows = [
            {
                "condition": "base",
                "prefix_mode": "hf",
                "example_id": 1,
                "step": 0,
                "top1_equal": True,
                "topk_overlap": 2,
                "hf_top1_token_id": 10,
                "vllm_generated_token_id": 10,
                "max_common_abs_logprob_delta": 0.1,
            },
            {
                "condition": "base",
                "prefix_mode": "hf",
                "example_id": 1,
                "step": 1,
                "top1_equal": False,
                "topk_overlap": 1,
                "hf_top1_token_id": 11,
                "vllm_generated_token_id": 12,
                "max_common_abs_logprob_delta": 0.2,
            },
        ]

        summary = summarize(rows)

        self.assertEqual(summary["kind"], "backend_token_replay_probe")
        self.assertAlmostEqual(summary["overall_top1_equal_rate"], 0.5)
        condition = summary["conditions"]["base|hf"]
        self.assertAlmostEqual(condition["top1_equal_rate"], 0.5)
        self.assertAlmostEqual(condition["mean_topk_overlap"], 1.5)
        self.assertAlmostEqual(condition["first_mismatch_rate"], 1.0)
        self.assertEqual(condition["mean_first_mismatch_step"], 1)


if __name__ == "__main__":
    unittest.main()
