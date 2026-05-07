import json
import tempfile
import unittest
from pathlib import Path

from randopt_lora_lab.backend_output_diff import compare_rows, main


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


class BackendOutputDiffTests(unittest.TestCase):
    def test_compare_rows_groups_candidate_deltas(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trusted = root / "trusted"
            candidate = root / "candidate"
            trusted.mkdir()
            candidate.mkdir()
            write_jsonl(
                trusted / "per_prompt.jsonl",
                [
                    {"mode": "screen", "candidate": "a", "example_id": 1, "exact": 1.0, "answer": "1+1", "text": "<answer>1+1</answer>"},
                    {"mode": "screen", "candidate": "a", "example_id": 2, "exact": 0.0, "answer": "bad", "text": "<answer>bad</answer>"},
                    {"mode": "base_screen", "candidate": "base", "example_id": 1, "exact": 0.0},
                ],
            )
            write_jsonl(
                candidate / "per_prompt.jsonl",
                [
                    {"mode": "screen", "candidate": "a", "example_id": 1, "exact": 0.0, "answer": "2", "text": "<answer>2</answer>"},
                    {"mode": "screen", "candidate": "a", "example_id": 2, "exact": 0.0, "answer": "bad", "text": "<answer>bad</answer>"},
                ],
            )
            detail, by_candidate, summary = compare_rows(trusted, candidate, mode="screen", trusted_name="peft", candidate_name="vllm")
            self.assertEqual(len(detail), 2)
            self.assertEqual(len(by_candidate), 1)
            self.assertAlmostEqual(by_candidate[0]["exact_delta"], -0.5)
            self.assertAlmostEqual(summary["exact_disagreement_rate"], 0.5)
            self.assertAlmostEqual(summary["answer_equal_rate"], 0.5)

    def test_cli_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trusted = root / "trusted"
            candidate = root / "candidate"
            out = root / "out"
            trusted.mkdir()
            candidate.mkdir()
            row = {"mode": "screen", "candidate": "a", "example_id": 1, "exact": 1.0, "answer": "x", "text": "x"}
            write_jsonl(trusted / "per_prompt.jsonl", [row])
            write_jsonl(candidate / "per_prompt.jsonl", [row])
            rc = main(["--trusted", str(trusted), "--candidate", str(candidate), "--out", str(out)])
            self.assertEqual(rc, 0)
            self.assertTrue((out / "summary.json").exists())
            self.assertTrue((out / "candidate_diff.csv").exists())


if __name__ == "__main__":
    unittest.main()
