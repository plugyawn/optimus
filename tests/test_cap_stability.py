import unittest

from randopt_lora_lab.cap_stability import (
    compact_tagged_prompt,
    direct_tagged_prompt,
    metric_row,
    prompt_fn,
    reordered_tagged_prompt,
    tight_tagged_prompt,
    xml_tagged_prompt,
)
from randopt_lora_lab.countdown import CountdownExample


class CapStabilityTests(unittest.TestCase):
    def test_tight_prompt_keeps_tagged_answer_contract(self):
        ex = CountdownExample(1, (3, 7, 8, 8), 24)
        text = tight_tagged_prompt(ex)

        self.assertIn("<answer>EXPRESSION</answer>", text)
        self.assertIn("No reasoning", text)
        self.assertIn("Target: 24", text)

    def test_compact_and_direct_prompts_keep_numbers_and_tags(self):
        ex = CountdownExample(1, (3, 7, 8, 8), 24)

        for text in [
            compact_tagged_prompt(ex),
            direct_tagged_prompt(ex),
            reordered_tagged_prompt(ex),
            xml_tagged_prompt(ex),
        ]:
            self.assertIn("<answer>", text)
            self.assertIn("</answer>", text)
            self.assertIn("24", text)
            self.assertIn("3", text)

    def test_reordered_and_xml_prompts_preserve_full_default_contract(self):
        ex = CountdownExample(1, (3, 7, 8, 8), 24)

        for text in [reordered_tagged_prompt(ex), xml_tagged_prompt(ex)]:
            self.assertIn("exactly once", text)
            self.assertIn("equals sign", text)
            self.assertIn("reasoning", text)
            self.assertIn("any other text", text)

    def test_prompt_fn_rejects_unknown_variant(self):
        with self.assertRaises(ValueError):
            prompt_fn("unknown")

    def test_metric_row_keeps_audit_dimensions(self):
        ev = {
            "candidate": "c",
            "exact_mean": 0.25,
            "malformed_mean": 0.125,
            "cap_hit_mean": 0.5,
            "answer_closed_mean": 0.75,
            "output_tokens": 10,
            "output_token_mean": 2.5,
            "output_token_p95": 4.0,
            "elapsed_s": 1.2,
            "mutation_s": 0.1,
        }

        row = metric_row(ev, cap=64, prompt_variant="tight", split="holdout", candidate_kind="elite_0")

        self.assertEqual(row["max_new_tokens"], 64)
        self.assertEqual(row["prompt_variant"], "tight")
        self.assertEqual(row["split"], "holdout")
        self.assertEqual(row["candidate_kind"], "elite_0")
        self.assertEqual(row["exact_mean"], 0.25)


if __name__ == "__main__":
    unittest.main()
