import unittest

from randopt_lora_lab.prompt_robustness import gate_prompt_robustness, summarize_rows


class PromptRobustnessTests(unittest.TestCase):
    def test_summarize_marks_base_collapsed_prompt_invalid(self):
        rows = [
            {"split": "holdout", "prompt_variant": "good", "max_new_tokens": 64, "candidate_kind": "base", "exact_mean": 0.1, "malformed_mean": 0.0, "cap_hit_mean": 0.0},
            {"split": "holdout", "prompt_variant": "good", "max_new_tokens": 64, "candidate_kind": "aggregate", "exact_mean": 0.2, "malformed_mean": 0.0, "cap_hit_mean": 0.0},
            {"split": "holdout", "prompt_variant": "bad", "max_new_tokens": 64, "candidate_kind": "base", "exact_mean": 0.0, "malformed_mean": 0.9, "cap_hit_mean": 0.0},
            {"split": "holdout", "prompt_variant": "bad", "max_new_tokens": 64, "candidate_kind": "aggregate", "exact_mean": 0.1, "malformed_mean": 0.0, "cap_hit_mean": 0.0},
        ]

        lifted, prompt_caps = summarize_rows(
            rows,
            split="holdout",
            max_base_malformed=0.05,
            max_base_cap_hit=0.05,
            target_kind="aggregate",
        )

        by_prompt = {row["prompt_variant"]: row for row in lifted if row["candidate_kind"] == "aggregate"}
        self.assertTrue(by_prompt["good"]["protocol_valid"])
        self.assertFalse(by_prompt["bad"]["protocol_valid"])
        self.assertAlmostEqual(by_prompt["good"]["lift_vs_base"], 0.1)
        self.assertAlmostEqual(by_prompt["good"]["malformed_regression_vs_base"], 0.0)
        self.assertAlmostEqual(by_prompt["good"]["cap_hit_regression_vs_base"], 0.0)
        self.assertEqual(len(prompt_caps), 2)

    def test_gate_requires_multiple_valid_prompt_variants(self):
        prompt_caps = {
            ("good", 64): {"protocol_valid": True, "lift_vs_base": 0.1},
            ("good", 128): {"protocol_valid": True, "lift_vs_base": 0.1},
            ("bad", 64): {"protocol_valid": False, "lift_vs_base": 0.1},
        }
        for row in prompt_caps.values():
            row.update(
                {
                    "malformed_mean": 0.0,
                    "cap_hit_mean": 0.0,
                    "malformed_regression_vs_base": 0.0,
                    "cap_hit_regression_vs_base": 0.0,
                }
            )

        gate = gate_prompt_robustness(
            prompt_caps,
            min_valid_prompts=2,
            min_lift=0.0,
            max_candidate_malformed=0.05,
            max_candidate_cap_hit=0.05,
            max_malformed_regression=0.05,
            max_cap_hit_regression=0.05,
        )

        self.assertFalse(gate["pass"])
        self.assertEqual(gate["valid_prompt_conditions"], 2)
        self.assertEqual(gate["valid_prompt_variants"], 1)
        self.assertEqual(gate["passing_prompt_variants"], 1)

    def test_gate_rejects_prompt_with_cap_hit_regression(self):
        prompt_caps = {
            ("good", 64): {
                "protocol_valid": True,
                "lift_vs_base": 0.1,
                "malformed_mean": 0.0,
                "cap_hit_mean": 0.2,
                "malformed_regression_vs_base": 0.0,
                "cap_hit_regression_vs_base": 0.2,
            },
            ("other", 64): {
                "protocol_valid": True,
                "lift_vs_base": 0.1,
                "malformed_mean": 0.0,
                "cap_hit_mean": 0.0,
                "malformed_regression_vs_base": 0.0,
                "cap_hit_regression_vs_base": 0.0,
            },
        }

        gate = gate_prompt_robustness(
            prompt_caps,
            min_valid_prompts=2,
            min_lift=0.0,
            max_candidate_malformed=0.05,
            max_candidate_cap_hit=0.05,
            max_malformed_regression=0.05,
            max_cap_hit_regression=0.05,
        )

        self.assertFalse(gate["pass"])
        self.assertEqual(gate["passing_prompt_variants"], 1)


if __name__ == "__main__":
    unittest.main()
