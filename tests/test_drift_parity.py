import json
import tempfile
import unittest
from pathlib import Path

from randopt_lora_lab.drift_parity import run_drift_parity


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def write_drift_run(path: Path, family: str, *, kl: float, l2: float, top1: float, prompts: int = 4) -> None:
    write_json(
        path / "summary.json",
        {
            "kind": "logit_drift",
            "family": family,
            "population": 2,
            "rank": 8,
            "sigma": 0.01,
            "sigma_values": [0.01],
            "prompts": prompts,
            "kl_base_to_candidate_mean_mean": kl,
            "kl_base_to_candidate_mean_max": kl,
            "kl_candidate_to_base_mean_mean": kl * 1.1,
            "logit_l2_mean_mean": l2,
            "top1_equal_rate_mean": top1,
            "top1_equal_rate_min": top1,
        },
    )
    write_jsonl(
        path / "candidate_drift.jsonl",
        [
            {
                "candidate": f"{family}:seed1:s0.01:sign1",
                "kl_base_to_candidate_mean": kl,
                "kl_candidate_to_base_mean": kl * 1.1,
                "logit_l2_mean": l2,
                "top1_equal_rate": top1,
            },
            {
                "candidate": f"{family}:seed2:s0.01:sign1",
                "kl_base_to_candidate_mean": kl,
                "kl_candidate_to_base_mean": kl * 1.1,
                "logit_l2_mean": l2,
                "top1_equal_rate": top1,
            },
        ],
    )


class DriftParityTests(unittest.TestCase):
    def test_passes_when_candidate_drift_is_no_worse_than_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "dense"
            candidate = root / "lora"
            write_drift_run(reference, "dense_gaussian", kl=0.02, l2=2.0, top1=0.95)
            write_drift_run(candidate, "factor_gaussian_lora", kl=0.018, l2=1.8, top1=0.95)

            summary = run_drift_parity(reference, candidate)

        self.assertTrue(summary["pass"])
        self.assertEqual(summary["failed"], [])
        self.assertLess(summary["comparison"]["kl_base_to_candidate_mean_ratio"], 1.0)

    def test_fails_negative_kl_and_drift_regression(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "dense"
            candidate = root / "lora"
            write_drift_run(reference, "dense_gaussian", kl=0.02, l2=2.0, top1=0.95)
            write_drift_run(candidate, "factor_gaussian_lora", kl=0.05, l2=3.0, top1=0.80)
            rows = (candidate / "candidate_drift.jsonl").read_text().splitlines()
            bad = json.loads(rows[0])
            bad["kl_base_to_candidate_mean"] = -0.01
            rows[0] = json.dumps(bad)
            (candidate / "candidate_drift.jsonl").write_text("\n".join(rows) + "\n")

            summary = run_drift_parity(reference, candidate)

        self.assertFalse(summary["pass"])
        self.assertIn("kl_nonnegative", summary["failed"])
        self.assertIn("candidate_mean_kl_not_higher", summary["failed"])
        self.assertIn("candidate_logit_l2_not_higher", summary["failed"])
        self.assertIn("candidate_top1_not_worse", summary["failed"])


if __name__ == "__main__":
    unittest.main()
