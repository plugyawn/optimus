import json
import tempfile
import unittest
from pathlib import Path

from randopt_lora_lab.proposal_audit import compare_summaries, proposal_gate, summarize_run


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def candidate(seed: int, sign: int, score: float, exact: float, cap: float = 0.0, malformed: float = 0.0) -> dict:
    return {
        "candidate": f"isotropic:seed{seed}:s0.01:sign{sign}",
        "seed": seed,
        "sigma": 0.01,
        "sign": sign,
        "selection_score": score,
        "exact_mean": exact,
        "cap_hit_mean": cap,
        "malformed_mean": malformed,
    }


def condition(seed: int, sign: int, variant: str, score: float, exact: float) -> dict:
    row = candidate(seed, sign, score, exact)
    row["prompt_variant"] = variant
    row["condition_selection_score"] = score
    return row


class ProposalAuditTests(unittest.TestCase):
    def make_run(self, root: Path, name: str, offset: float) -> Path:
        run = root / name
        write_json(
            run / "summary.json",
            {
                "population": 4,
                "base_screen_exact": 0.1,
                "base_holdout_exact": 0.2,
                "candidate_sec": 2.0 + offset,
                "screen_candidate_sec": 3.0 + offset,
                "ensemble_holdout": [{"k": 2, "exact_mean": 0.4 + offset}],
            },
        )
        write_jsonl(
            run / "candidate_summary.jsonl",
            [
                candidate(1, 1, 0.30 + offset, 0.25),
                candidate(1, -1, -0.10, 0.05, malformed=0.5),
                candidate(2, 1, 0.20, 0.15),
                candidate(2, -1, 0.00, 0.10),
            ],
        )
        write_jsonl(
            run / "holdout_candidate_summary.jsonl",
            [
                candidate(1, 1, 0.10, 0.35),
                candidate(2, 1, 0.20, 0.30),
            ],
        )
        write_jsonl(
            run / "candidate_condition_summary.jsonl",
            [
                condition(1, 1, "default", 0.30 + offset, 0.25),
                condition(1, 1, "reordered", 0.25 + offset, 0.20),
                condition(1, -1, "default", -0.10, 0.05),
                condition(1, -1, "reordered", -0.20, 0.00),
                condition(2, 1, "default", 0.20, 0.15),
                condition(2, 1, "reordered", 0.10, 0.10),
                condition(2, -1, "default", 0.00, 0.10),
                condition(2, -1, "reordered", -0.05, 0.05),
            ],
        )
        return run

    def test_summarize_run_reports_transfer_prompt_and_pair_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make_run(Path(tmp), "run", 0.0)
            summary = summarize_run(run, top_k=2)

        self.assertEqual(summary["screen_candidates"], 4)
        self.assertEqual(summary["best_ensemble_holdout_exact"], 0.4)
        self.assertEqual(summary["prompt_variants"]["common_candidates"], 4)
        self.assertEqual(summary["prompt_variants"]["top2_selection_overlap"], 2)
        self.assertEqual(summary["holdout_transfer"]["common_candidates"], 2)
        self.assertAlmostEqual(summary["holdout_transfer"]["best_holdout_exact"], 0.35)
        self.assertEqual(summary["antithetic_pairs"]["pairs"], 2)
        self.assertGreater(summary["antithetic_pairs"]["pair_score_gap_mean"], 0.0)

    def test_compare_summaries_reports_delta(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left = summarize_run(self.make_run(root, "left", 0.1), top_k=2)
            right = summarize_run(self.make_run(root, "right", 0.0), top_k=2)
            comparison = compare_summaries(left, right, left_name="left", right_name="right", top_k=2)

        self.assertAlmostEqual(comparison["delta"]["candidate_sec_left_minus_right"], 0.1)
        self.assertAlmostEqual(comparison["delta"]["best_ensemble_holdout_exact_left_minus_right"], 0.1)

    def test_gate_fails_slower_prompt_brittle_proposal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left_run = self.make_run(root, "left", -0.5)
            write_jsonl(
                left_run / "candidate_condition_summary.jsonl",
                [
                    condition(1, 1, "default", 0.30, 0.25),
                    condition(1, 1, "reordered", -0.30, 0.00),
                    condition(1, -1, "default", 0.20, 0.20),
                    condition(1, -1, "reordered", -0.20, 0.05),
                    condition(2, 1, "default", -0.20, 0.05),
                    condition(2, 1, "reordered", 0.20, 0.20),
                    condition(2, -1, "default", -0.30, 0.00),
                    condition(2, -1, "reordered", 0.30, 0.25),
                ],
            )
            left = summarize_run(left_run, top_k=2)
            right = summarize_run(self.make_run(root, "right", 0.0), top_k=2)
            comparison = compare_summaries(left, right, left_name="left", right_name="right", top_k=2)
            gate = proposal_gate(comparison, top_k=2, min_prompt_selection_spearman=0.99)

        self.assertFalse(gate["pass"])
        self.assertIn("candidate_throughput_not_slower", gate["failed"])
        self.assertIn("ensemble_quality_not_worse", gate["failed"])
        self.assertIn("prompt_selection_rank_stable", gate["failed"])


if __name__ == "__main__":
    unittest.main()
