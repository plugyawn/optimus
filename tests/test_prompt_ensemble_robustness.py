import unittest

from randopt_lora_lab.prompt_ensemble_robustness import gate_rows, summarize_ensemble_robustness


def summary_row(prompt, cap, kind, exact=0.0, malformed=0.0, cap_hit=0.0):
    return {
        "split": "holdout",
        "prompt_variant": prompt,
        "max_new_tokens": cap,
        "candidate_kind": kind,
        "exact_mean": exact,
        "malformed_mean": malformed,
        "cap_hit_mean": cap_hit,
    }


def prompt_row(prompt, cap, kind, example_id, text):
    return {
        "split": "holdout",
        "prompt_variant": prompt,
        "max_new_tokens": cap,
        "candidate_kind": kind,
        "example_id": example_id,
        "numbers": [1, 2, 3, 4],
        "target": 10,
        "text": text,
    }


class PromptEnsembleRobustnessTests(unittest.TestCase):
    def test_strict_topk_ensemble_lift_passes_across_prompts(self):
        summary_rows = []
        per_prompt_rows = []
        for prompt in ["default", "reordered"]:
            summary_rows.extend(
                [
                    summary_row(prompt, 64, "base", exact=0.0),
                    summary_row(prompt, 64, "elite_0", exact=1.0),
                    summary_row(prompt, 64, "elite_1", exact=1.0),
                ]
            )
            for example_id in [1, 2]:
                per_prompt_rows.extend(
                    [
                        prompt_row(prompt, 64, "elite_0", example_id, "<answer>1+2+3+4</answer>"),
                        prompt_row(prompt, 64, "elite_1", example_id, "<answer>1+2+3+4</answer>"),
                    ]
                )

        rows = summarize_ensemble_robustness(
            summary_rows,
            per_prompt_rows,
            split="holdout",
            k=2,
            strict_rows=True,
            max_base_malformed=0.05,
            max_base_cap_hit=0.05,
        )
        gate = gate_rows(
            rows,
            min_valid_prompts=2,
            min_lift=0.5,
            max_candidate_malformed=0.05,
            max_candidate_cap_hit=0.05,
            max_malformed_regression=0.05,
            max_cap_hit_regression=0.05,
        )

        self.assertEqual(len(rows), 2)
        self.assertTrue(gate["pass"])
        self.assertEqual(gate["passing_prompt_variants"], 2)
        self.assertEqual(rows[0]["correct"], 2)

    def test_gate_rejects_high_selected_malformed_even_with_lift(self):
        rows = [
            {
                "prompt_variant": "default",
                "protocol_valid": True,
                "lift_vs_base": 0.1,
                "max_candidate_malformed": 0.2,
                "max_candidate_cap_hit": 0.0,
                "max_malformed_regression_vs_base": 0.2,
                "max_cap_hit_regression_vs_base": 0.0,
            },
            {
                "prompt_variant": "reordered",
                "protocol_valid": True,
                "lift_vs_base": 0.1,
                "max_candidate_malformed": 0.0,
                "max_candidate_cap_hit": 0.0,
                "max_malformed_regression_vs_base": 0.0,
                "max_cap_hit_regression_vs_base": 0.0,
            },
        ]

        gate = gate_rows(
            rows,
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
