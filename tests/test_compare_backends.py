import tempfile
import unittest
from pathlib import Path

from optimus.evaluation.compare import average_ranks, compare, parse_ks, pearson, spearman


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(__import__("json").dumps(row) + "\n")


class CompareBackendsTests(unittest.TestCase):
    def test_average_ranks_handles_ties(self):
        self.assertEqual(average_ranks([0.2, 0.1, 0.2]), [2.5, 1.0, 2.5])

    def test_correlations(self):
        self.assertAlmostEqual(pearson([1, 2, 3], [1, 2, 3]), 1.0)
        self.assertAlmostEqual(spearman([10, 20, 30], [1, 2, 3]), 1.0)
        self.assertAlmostEqual(spearman([10, 20, 30], [3, 2, 1]), -1.0)

    def test_parse_ks(self):
        self.assertEqual(parse_ks("4,8, 16"), [4, 8, 16])

    def test_compare_reports_regret_and_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trusted = root / "trusted"
            candidate = root / "candidate"
            trusted.mkdir()
            candidate.mkdir()
            write_jsonl(
                trusted / "candidate_summary.jsonl",
                [
                    {"candidate": "a", "exact_mean": 0.30},
                    {"candidate": "b", "exact_mean": 0.20},
                    {"candidate": "c", "exact_mean": 0.10},
                ],
            )
            write_jsonl(
                candidate / "candidate_summary.jsonl",
                [
                    {"candidate": "a", "exact_mean": 0.10},
                    {"candidate": "b", "exact_mean": 0.20},
                    {"candidate": "c", "exact_mean": 0.30},
                ],
            )
            _, summary = compare(
                trusted,
                candidate,
                trusted_name="peft",
                candidate_name="vllm",
                ks=[2],
                spearman_gate=0.85,
                top8_gate=6,
            )
        self.assertEqual(summary["n_common"], 3)
        self.assertEqual(summary["top2_overlap"], 1)
        self.assertEqual(summary["candidate_best_candidate"], "c")
        self.assertAlmostEqual(summary["selected_regret_vs_trusted"], 0.20)
        self.assertFalse(summary["pass_spearman_gate"])


if __name__ == "__main__":
    unittest.main()
