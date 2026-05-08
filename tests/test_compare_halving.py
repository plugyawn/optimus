import json
import tempfile
import unittest
from pathlib import Path

from randopt_lora_lab.compare_halving import compare_halving
from randopt_lora_lab.vllm_lora_halving import build_parser


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


class CompareHalvingTests(unittest.TestCase):
    def test_vllm_halving_parser_has_shared_prompt_input_arg(self):
        args = build_parser().parse_args(["--out", "tmp"])

        self.assertEqual(args.prompt_input, "text")

    def test_reports_survivor_recall_and_regret(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full = root / "full"
            halving = root / "halving"
            full.mkdir()
            halving.mkdir()
            write_jsonl(
                full / "candidate_summary.jsonl",
                [
                    {"candidate": "a", "exact_mean": 0.40},
                    {"candidate": "b", "exact_mean": 0.30},
                    {"candidate": "c", "exact_mean": 0.20},
                    {"candidate": "d", "exact_mean": 0.10},
                ],
            )
            write_jsonl(
                halving / "stage_candidate_summary.jsonl",
                [
                    {"candidate": "a", "exact_mean": 0.00},
                    {"candidate": "b", "exact_mean": 0.50},
                    {"candidate": "c", "exact_mean": 0.40},
                    {"candidate": "d", "exact_mean": 0.10},
                ],
            )
            write_jsonl(
                halving / "candidate_summary.jsonl",
                [
                    {"candidate": "b", "exact_mean": 0.30},
                    {"candidate": "c", "exact_mean": 0.20},
                ],
            )
            _, summary = compare_halving(full, halving, ks=[1, 2])
        self.assertFalse(summary["full_best_survived"])
        self.assertEqual(summary["top1_survivor_recall"], 0)
        self.assertEqual(summary["top2_survivor_recall"], 1)
        self.assertEqual(summary["halving_best_candidate"], "b")
        self.assertAlmostEqual(summary["halving_selected_regret_vs_full"], 0.10)


if __name__ == "__main__":
    unittest.main()
