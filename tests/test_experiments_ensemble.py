import unittest

from randopt_lora_lab.countdown import CountdownExample
from randopt_lora_lab.experiments import candidate_panel, ensemble_ks_from_values, majority_vote_evaluation, parse_k_list
from randopt_lora_lab.vllm_lora_search import candidate_panel as vllm_candidate_panel, read_candidate_file


class ExperimentEnsembleTests(unittest.TestCase):
    def test_parse_k_list_sorts_and_deduplicates(self):
        self.assertEqual(parse_k_list("4,1,4,2"), [1, 2, 4])

    def test_ensemble_ks_can_be_derived_from_paper_ratios(self):
        self.assertEqual(ensemble_ks_from_values(5000, ratio_text="0.04,0.01,0.05,0.1"), [50, 200, 250, 500])
        self.assertEqual(ensemble_ks_from_values(128, k_text="8,16", ratio_text="0.04,0.01"), [1, 5, 8, 16])

    def test_candidate_panel_samples_only_requested_sigmas(self):
        candidates = candidate_panel(
            "dense_gaussian",
            population=16,
            sigma=0.01,
            seed=123,
            antithetic=False,
            sigma_values=[0.0005, 0.001, 0.002],
        )

        self.assertEqual(len(candidates), 16)
        self.assertTrue({c.sigma for c in candidates}.issubset({0.0005, 0.001, 0.002}))
        self.assertGreater(len({c.sigma for c in candidates}), 1)

    def test_vllm_candidate_panel_uses_same_multi_sigma_semantics(self):
        candidates = vllm_candidate_panel(
            "isotropic",
            population=16,
            sigma=0.01,
            seed=123,
            antithetic=False,
            sigma_values=[0.0005, 0.001, 0.002],
        )

        self.assertEqual(len(candidates), 16)
        self.assertEqual([c.sigma for c in candidates], [c.sigma for c in candidate_panel(
            "isotropic",
            population=16,
            sigma=0.01,
            seed=123,
            antithetic=False,
            sigma_values=[0.0005, 0.001, 0.002],
        )])

    def test_vllm_candidate_file_reads_jsonl_candidate_keys(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "candidates.jsonl"
            path.write_text(
                '{"candidate":"isotropic:seed7:s0.01:sign1"}\n'
                "isotropic:seed8:s0.02:sign-1\n"
            )
            candidates = read_candidate_file(str(path))

        self.assertEqual([c.seed for c in candidates], [7, 8])
        self.assertEqual([c.sign for c in candidates], [1, -1])
        self.assertEqual([c.sigma for c in candidates], [0.01, 0.02])

    def test_majority_vote_uses_numeric_answers_not_formula_strings(self):
        example = CountdownExample(1, (1, 2, 3), 6)
        rows = [
            {"candidate": "c1", "example_id": 1, "text": "<answer>1+2+3</answer>"},
            {"candidate": "c2", "example_id": 1, "text": "<answer>3*2*1</answer>"},
            {"candidate": "c3", "example_id": 1, "text": "<answer>1+2-3</answer>"},
        ]

        summary, per_prompt = majority_vote_evaluation(["c1", "c2", "c3"], rows, [example], [1, 2, 3])

        self.assertEqual([row["exact_mean"] for row in summary], [1.0, 1.0, 1.0])
        self.assertEqual(per_prompt[1]["vote_counts"], {"6": 2})
        self.assertEqual(per_prompt[2]["vote_counts"], {"6": 2, "0": 1})

    def test_majority_vote_rejects_invalid_formulas(self):
        example = CountdownExample(1, (1, 2, 3), 6)
        rows = [
            {"candidate": "c1", "example_id": 1, "text": "<answer>6</answer>"},
            {"candidate": "c2", "example_id": 1, "text": "no answer"},
        ]

        summary, per_prompt = majority_vote_evaluation(["c1", "c2"], rows, [example], [2])

        self.assertEqual(summary[0]["exact_mean"], 0.0)
        self.assertEqual(summary[0]["coverage_mean"], 0.0)
        self.assertEqual(per_prompt[0]["reject_counts"], {"wrong_numbers": 1, "missing_answer": 1})


if __name__ == "__main__":
    unittest.main()
