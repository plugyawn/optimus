import json
import tempfile
import unittest
from pathlib import Path

from randopt_lora_lab.parity_report import candidate_spec_key, compare_runs, load_run


def write_run(path: Path, family: str, scores: list[float], candidate_sec: float):
    path.mkdir(parents=True)
    (path / "summary.json").write_text(json.dumps({"candidate_sec": candidate_sec}) + "\n")
    with (path / "candidate_summary.jsonl").open("w") as f:
        for idx, score in enumerate(scores):
            row = {
                "candidate": f"{family}:seed{idx + 1}:s0.01:sign1",
                "exact_mean": score,
            }
            f.write(json.dumps(row) + "\n")


class ParityReportTests(unittest.TestCase):
    def test_candidate_spec_key_ignores_family(self):
        self.assertEqual(candidate_spec_key("dense_gaussian:seed7:s0.01:sign-1"), "seed7:s0.01:sign-1")
        self.assertEqual(candidate_spec_key("factor_gaussian_lora:seed7:s0.01:sign-1"), "seed7:s0.01:sign-1")

    def test_compare_runs_passes_matching_rankings_and_speedup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_run(root / "dense", "dense_gaussian", [0.4, 0.3, 0.2, 0.1], candidate_sec=1.0)
            write_run(root / "lora", "factor_gaussian_lora", [0.4, 0.3, 0.2, 0.1], candidate_sec=4.0)

            summary = compare_runs(load_run(root / "dense"), load_run(root / "lora"), top_k=2, min_topk_overlap=2)

            self.assertTrue(summary["pass"])
            self.assertEqual(summary["topk_overlap"], 2)
            self.assertEqual(summary["selected_regret"], 0.0)
            self.assertEqual(summary["speed_ratio_lora_over_dense"], 4.0)

    def test_compare_runs_fails_when_lora_picks_dense_loser(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_run(root / "dense", "dense_gaussian", [0.4, 0.3, 0.2, 0.1], candidate_sec=1.0)
            write_run(root / "lora", "factor_gaussian_lora", [0.1, 0.2, 0.3, 0.4], candidate_sec=4.0)

            summary = compare_runs(load_run(root / "dense"), load_run(root / "lora"), top_k=2, min_topk_overlap=2)

            self.assertFalse(summary["pass"])
            self.assertGreater(summary["selected_regret"], 0.0)
            self.assertFalse(summary["gates"]["spearman"])


if __name__ == "__main__":
    unittest.main()
