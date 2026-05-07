import json
import tempfile
import unittest
from pathlib import Path

from randopt_lora_lab.confirmation_economics import analyze, confirmation_gate


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


class ConfirmationEconomicsTests(unittest.TestCase):
    def test_reports_zero_regret_k_and_speedups(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trusted = root / "trusted"
            proposal = root / "proposal"
            trusted.mkdir()
            proposal.mkdir()
            write_json(trusted / "summary.json", {"population": 4, "candidate_sec": 1.0})
            write_json(proposal / "summary.json", {"population": 4, "eval_elapsed_s": 1.0, "load_s": 2.0, "adapter_build_s": 0.5})
            write_jsonl(
                trusted / "candidate_summary.jsonl",
                [
                    {"candidate": "a", "exact_mean": 0.3, "elapsed_s": 1.0},
                    {"candidate": "b", "exact_mean": 0.2, "elapsed_s": 1.0},
                    {"candidate": "c", "exact_mean": 0.1, "elapsed_s": 1.0},
                    {"candidate": "d", "exact_mean": 0.0, "elapsed_s": 1.0},
                ],
            )
            write_jsonl(
                proposal / "candidate_summary.jsonl",
                [
                    {"candidate": "b", "exact_mean": 0.5},
                    {"candidate": "a", "exact_mean": 0.4},
                    {"candidate": "c", "exact_mean": 0.1},
                    {"candidate": "d", "exact_mean": 0.0},
                ],
            )

            rows, summary = analyze(trusted, proposal, ks=[1, 2, 4])
            gate = confirmation_gate(rows, summary, max_confirm_k=2, min_eval_only_speedup=1.0, min_full_without_load_speedup=0.7)

            self.assertEqual(summary["best_recovered_k"], 2)
            self.assertEqual(summary["zero_regret_k"], 2)
            self.assertFalse(rows[0]["contains_trusted_best"])
            self.assertEqual(rows[0]["regret_vs_trusted_best"], 0.09999999999999998)
            self.assertTrue(rows[1]["contains_trusted_best"])
            self.assertEqual(rows[1]["confirmed_candidate"], "a")
            self.assertAlmostEqual(rows[1]["eval_only_speedup_vs_trusted_full"], 4 / 3)
            self.assertAlmostEqual(rows[1]["full_without_peft_load_speedup_vs_trusted_full"], 4 / 5.5)
            self.assertTrue(gate["pass"])

    def test_gate_fails_when_best_requires_too_large_k_or_speed_regresses(self):
        rows = [
            {
                "k": 1,
                "contains_trusted_best": False,
                "regret_vs_trusted_best": 0.1,
                "eval_only_speedup_vs_trusted_full": 2.0,
                "full_without_peft_load_speedup_vs_trusted_full": 2.0,
            },
            {
                "k": 4,
                "contains_trusted_best": True,
                "regret_vs_trusted_best": 0.0,
                "eval_only_speedup_vs_trusted_full": 0.8,
                "full_without_peft_load_speedup_vs_trusted_full": 0.7,
            },
        ]
        summary = {"best_recovered_k": 4, "zero_regret_k": 4}

        gate = confirmation_gate(rows, summary, max_confirm_k=2, min_eval_only_speedup=1.0)

        self.assertFalse(gate["pass"])
        self.assertIn("trusted_best_recovered_within_k", gate["failed"])
        self.assertIn("zero_regret_within_k", gate["failed"])
        self.assertIn("eval_only_speedup", gate["failed"])
        self.assertIn("full_without_peft_load_speedup", gate["failed"])


if __name__ == "__main__":
    unittest.main()
