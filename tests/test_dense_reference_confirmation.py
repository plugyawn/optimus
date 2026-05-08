import json
import tempfile
import unittest
from pathlib import Path

from randopt_lora_lab.dense_reference_confirmation import analyze, gate


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def run_dir(root: Path, name: str, family: str, scores: list[float], *, candidate_sec: float = 1.0) -> Path:
    path = root / name
    write_json(path / "summary.json", {"population": len(scores), "candidate_sec": candidate_sec})
    write_jsonl(
        path / "candidate_summary.jsonl",
        [
            {
                "candidate": f"{family}:seed{idx + 1}:s0.001:sign1",
                "exact_mean": score,
                "elapsed_s": 1.0,
            }
            for idx, score in enumerate(scores)
        ],
    )
    return path


class DenseReferenceConfirmationTests(unittest.TestCase):
    def test_zero_dense_regret_can_be_recovered_by_spectral_confirmed_shortlist(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dense = run_dir(root, "dense", "dense_gaussian", [0.4, 0.3, 0.2, 0.1], candidate_sec=1.0)
            spectral = run_dir(root, "spectral", "spectral_projected_gaussian_rank_r_c1p5", [0.2, 0.4, 0.1, 0.0], candidate_sec=1.0)
            proposal = run_dir(root, "proposal", "spectral_projected_gaussian_rank_r_c1p5", [0.1, 0.9, 0.8, 0.0], candidate_sec=4.0)
            write_json(proposal / "summary.json", {"population": 4, "eval_elapsed_s": 1.0, "load_s": 0.0, "adapter_build_s": 0.0})

            rows, summary = analyze(dense, spectral, proposal, ks=[1, 2, 4])
            result = gate(rows, summary, max_confirm_k=4, min_full_without_dense_load_speedup=1.0)

            self.assertEqual(summary["dense_best_spec"], "seed1:s0.001:sign1")
            self.assertEqual(rows[0]["confirmed_spec"], "seed2:s0.001:sign1")
            self.assertGreater(rows[0]["dense_regret_vs_best"], 0.0)
            self.assertEqual(rows[2]["confirmed_spec"], "seed2:s0.001:sign1")
            self.assertGreater(rows[2]["dense_regret_vs_best"], 0.0)
            self.assertFalse(result["pass"])

    def test_gate_passes_when_shortlist_confirms_dense_tied_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dense = run_dir(root, "dense", "dense_gaussian", [0.4, 0.4, 0.2, 0.1], candidate_sec=2.0)
            spectral = run_dir(root, "spectral", "spectral_projected_gaussian_rank_r_c1p5", [0.2, 0.4, 0.1, 0.0], candidate_sec=1.0)
            proposal = run_dir(root, "proposal", "spectral_projected_gaussian_rank_r_c1p5", [0.1, 0.9, 0.8, 0.0], candidate_sec=8.0)
            write_json(proposal / "summary.json", {"population": 4, "eval_elapsed_s": 0.5, "load_s": 0.0, "adapter_build_s": 0.0})

            rows, summary = analyze(dense, spectral, proposal, ks=[1, 2])
            result = gate(rows, summary, max_confirm_k=1, min_full_without_dense_load_speedup=1.0)

            self.assertEqual(summary["zero_dense_regret_k"], 1)
            self.assertEqual(rows[0]["confirmed_spec"], "seed2:s0.001:sign1")
            self.assertEqual(rows[0]["dense_regret_vs_best"], 0.0)
            self.assertTrue(result["pass"])


if __name__ == "__main__":
    unittest.main()
