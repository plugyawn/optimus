import unittest

from randopt_lora_lab.selection_score import combine_candidate_conditions, enrich_condition_rows, parse_prompt_variants


class SelectionScoreTests(unittest.TestCase):
    def test_parse_prompt_variants_defaults(self):
        self.assertEqual(parse_prompt_variants(""), ["default"])
        self.assertEqual(parse_prompt_variants("default,reordered"), ["default", "reordered"])

    def test_robust_min_rejects_prompt_specific_candidate(self):
        base = {
            "default": {"exact_mean": 0.1, "malformed_mean": 0.0, "cap_hit_mean": 0.0},
            "reordered": {"exact_mean": 0.05, "malformed_mean": 0.0, "cap_hit_mean": 0.0},
        }
        rows = [
            {"candidate": "good", "prompt_variant": "default", "exact_mean": 0.15, "malformed_mean": 0.0, "cap_hit_mean": 0.0},
            {"candidate": "good", "prompt_variant": "reordered", "exact_mean": 0.08, "malformed_mean": 0.0, "cap_hit_mean": 0.0},
            {"candidate": "brittle", "prompt_variant": "default", "exact_mean": 0.2, "malformed_mean": 0.0, "cap_hit_mean": 0.0},
            {"candidate": "brittle", "prompt_variant": "reordered", "exact_mean": 0.0, "malformed_mean": 0.9, "cap_hit_mean": 0.5},
        ]

        combined = combine_candidate_conditions(
            rows,
            base,
            score_mode="robust_min",
            malformed_penalty=1.0,
            cap_hit_penalty=1.0,
        )
        by_candidate = {row["candidate"]: row for row in combined}

        self.assertGreater(by_candidate["good"]["selection_score"], by_candidate["brittle"]["selection_score"])
        self.assertAlmostEqual(by_candidate["good"]["min_exact_lift_vs_base"], 0.03)
        self.assertLess(by_candidate["brittle"]["min_condition_selection_score"], -1.0)

    def test_enrich_condition_rows_adds_regressions(self):
        enriched = enrich_condition_rows(
            [{"candidate": "x", "prompt_variant": "default", "exact_mean": 0.2, "malformed_mean": 0.1, "cap_hit_mean": 0.2}],
            {"default": {"exact_mean": 0.1, "malformed_mean": 0.0, "cap_hit_mean": 0.0}},
            malformed_penalty=2.0,
            cap_hit_penalty=3.0,
        )

        row = enriched[0]
        self.assertAlmostEqual(row["exact_lift_vs_base"], 0.1)
        self.assertAlmostEqual(row["malformed_regression_vs_base"], 0.1)
        self.assertAlmostEqual(row["cap_hit_regression_vs_base"], 0.2)
        self.assertAlmostEqual(row["condition_selection_score"], 0.1 - 2.0 * 0.1 - 3.0 * 0.2)


if __name__ == "__main__":
    unittest.main()
