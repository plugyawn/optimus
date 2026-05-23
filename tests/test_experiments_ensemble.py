import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from optimus.tasks.countdown import CountdownExample
from optimus.search.ensemble import ensemble_ks_from_values, majority_vote_evaluation, parse_k_list
from optimus.search.peft import (
    candidate_panel,
    maybe_build_family_state,
    read_candidate_file as read_peft_candidate_file,
    record_loaded_family_state,
)
from randopt_lora_lab.strict_ensemble_replay import replay
from optimus.core.candidates import candidate_panel as vllm_candidate_panel, read_candidate_file


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

    def test_peft_candidate_file_reads_same_jsonl_candidate_keys(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "candidates.jsonl"
            path.write_text(
                '{"candidate":"sparse_low_rank_lora_d0p125:seed7:s0.001:sign1"}\n'
                "sparse_low_rank_lora_d0p125:seed8:s0.002:sign-1\n"
            )
            candidates = read_peft_candidate_file(str(path))

        self.assertEqual([c.family for c in candidates], ["sparse_low_rank_lora_d0p125", "sparse_low_rank_lora_d0p125"])
        self.assertEqual([c.seed for c in candidates], [7, 8])
        self.assertEqual([c.sign for c in candidates], [1, -1])
        self.assertEqual([c.sigma for c in candidates], [0.001, 0.002])

    def test_peft_search_can_load_and_record_saved_family_state(self):
        import argparse
        import json
        import tempfile

        import torch

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            state_path = root / "family_state.pt"
            summary_path = root / "family_state_summary.json"
            torch.save({"module": torch.eye(2)}, state_path)
            summary_path.write_text(json.dumps({"activation_state_prompt_variants": ["default", "xml"]}))
            args = argparse.Namespace(
                family_state_file=str(state_path),
                family="activation_spectral_lora_c2",
                rank=32,
                targets="q_proj",
            )

            loaded = maybe_build_family_state(args, backend=None, screen=[])
            record_loaded_family_state(str(state_path), root / "confirmed", args)
            recorded = json.loads((root / "confirmed" / "family_state_summary.json").read_text())

        self.assertTrue(torch.equal(loaded["module"], torch.eye(2)))
        self.assertEqual(recorded["kind"], "loaded_family_state")
        self.assertEqual(recorded["source_summary"]["activation_state_prompt_variants"], ["default", "xml"])

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

    def test_strict_majority_vote_rejects_malformed_rows_before_voting(self):
        example = CountdownExample(1, (1, 2, 3), 6)
        rows = [
            {"candidate": "c1", "example_id": 1, "text": "reasoning <answer>1+2+3</answer>"},
            {"candidate": "c2", "example_id": 1, "text": "<answer>1+2+3</answer>"},
        ]

        lax, _ = majority_vote_evaluation(["c1", "c2"], rows, [example], [2])
        strict, per_prompt = majority_vote_evaluation(["c1", "c2"], rows, [example], [2], strict_rows=True)

        self.assertEqual(lax[0]["valid_votes_per_prompt"], 2.0)
        self.assertEqual(strict[0]["valid_votes_per_prompt"], 1.0)
        self.assertEqual(strict[0]["exact_mean"], 1.0)
        self.assertTrue(strict[0]["strict_rows"])
        self.assertEqual(per_prompt[0]["reject_counts"], {"strict_malformed": 1})

    def test_strict_ensemble_replay_reads_saved_run_rows(self):
        with TemporaryDirectory() as td:
            run = Path(td)
            (run / "summary.json").write_text(
                '{"ensemble_ks":[2],"top_screen":[{"candidate":"c1"},{"candidate":"c2"}]}\n'
            )
            (run / "holdout_per_prompt.jsonl").write_text(
                '{"candidate":"c1","example_id":1,"numbers":[1,2,3],"target":6,"text":"reasoning <answer>1+2+3</answer>"}\n'
                '{"candidate":"c2","example_id":1,"numbers":[1,2,3],"target":6,"text":"<answer>1+2+3</answer>"}\n'
            )

            payload = replay(run)

        self.assertEqual(payload["best_numeric_ensemble_holdout_exact"], 1.0)
        self.assertEqual(payload["best_strict_ensemble_holdout_exact"], 1.0)
        self.assertEqual(payload["strict_ensemble_holdout"][0]["valid_votes_per_prompt"], 1.0)


if __name__ == "__main__":
    unittest.main()
