import unittest

from optimus.search.selection import (
    combine_candidate_conditions,
    enrich_condition_rows,
    filter_condition_rows_by_variants,
    parse_prompt_variants,
    protocol_valid_variants,
)
from randopt_lora_lab.selection_score import combine_candidate_conditions as legacy_combine_candidate_conditions


class SelectionScoreTests(unittest.TestCase):
    def test_legacy_namespace_reexports_public_condition_combiner(self):
        self.assertIs(legacy_combine_candidate_conditions, combine_candidate_conditions)

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

    def test_protocol_valid_variants_excludes_collapsed_base_prompts(self):
        base = {
            "default": {"exact_mean": 0.1, "malformed_mean": 0.0, "cap_hit_mean": 0.0},
            "tight": {"exact_mean": 0.0, "malformed_mean": 0.8, "cap_hit_mean": 0.2},
            "capped": {"exact_mean": 0.05, "malformed_mean": 0.0, "cap_hit_mean": 0.2},
        }

        self.assertEqual(
            protocol_valid_variants(base, max_malformed=0.05, max_cap_hit=0.05),
            ["default"],
        )

    def test_filter_condition_rows_keeps_only_selector_variants(self):
        rows = [
            {"candidate": "x", "prompt_variant": "default"},
            {"candidate": "x", "prompt_variant": "tight"},
            {"candidate": "y", "prompt_variant": "default"},
        ]

        filtered = filter_condition_rows_by_variants(rows, ["default"])

        self.assertEqual([row["candidate"] for row in filtered], ["x", "y"])
        self.assertTrue(all(row["prompt_variant"] == "default" for row in filtered))


if __name__ == "__main__":
    unittest.main()
