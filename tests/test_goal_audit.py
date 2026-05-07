import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from randopt_lora_lab.goal_audit import run_goal_audit


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


class GoalAuditTests(unittest.TestCase):
    def test_missing_evidence_fails_every_goal_dimension(self):
        args = Namespace(
            reproduction_audit=None,
            parity_report=None,
            backend_gate=None,
            prompt_robustness=None,
            drift_report=None,
            adapter_run=None,
        )

        summary = run_goal_audit(args)

        self.assertFalse(summary["pass"])
        self.assertIn("official full-Gaussian baseline validity", summary["failed"])
        self.assertIn("quality parity", summary["failed"])
        self.assertIn("drift parity", summary["failed"])

    def test_full_evidence_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reproduction = root / "repro" / "summary.json"
            parity = root / "parity" / "summary.json"
            backend = root / "backend" / "summary.json"
            prompt = root / "prompt" / "summary.json"
            drift = root / "drift" / "summary.json"
            adapter = root / "adapter"
            write_json(reproduction, {"pass": True})
            write_json(
                parity,
                {
                    "pass": True,
                    "ensemble_holdout_delta_lora_minus_dense": 0.0,
                    "speed_ratio_lora_over_dense": 1.5,
                    "gates": {
                        "ensemble_quality": True,
                        "spearman": True,
                        "topk_overlap": True,
                        "selected_regret": True,
                        "speed": True,
                    },
                },
            )
            write_json(backend, {"pass": True})
            write_json(prompt, {"gate": {"pass": True, "valid_prompt_variants": 2, "passing_prompt_variants": 2, "min_valid_prompts": 2}})
            write_json(drift, {"pass": True})
            write_json(adapter / "summary.json", {"adapters_kept": True})
            args = Namespace(
                reproduction_audit=reproduction,
                parity_report=parity,
                backend_gate=backend,
                prompt_robustness=prompt,
                drift_report=drift,
                adapter_run=adapter,
            )

            summary = run_goal_audit(args)

            self.assertTrue(summary["pass"])
            self.assertEqual(summary["failed"], [])

    def test_parity_report_must_pass_stability_and_speed_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parity = root / "parity" / "summary.json"
            write_json(
                parity,
                {
                    "pass": False,
                    "spearman": 0.1,
                    "topk_overlap": 1,
                    "selected_regret": 0.2,
                    "speed_ratio_lora_over_dense": 0.5,
                    "gates": {
                        "ensemble_quality": True,
                        "spearman": False,
                        "topk_overlap": False,
                        "selected_regret": False,
                        "speed": False,
                    },
                },
            )
            args = Namespace(
                reproduction_audit=None,
                parity_report=parity,
                backend_gate=None,
                prompt_robustness=None,
                drift_report=None,
                adapter_run=None,
            )

            summary = run_goal_audit(args)

            self.assertIn("quality parity", summary["failed"])
            self.assertIn("stability parity", summary["failed"])
            self.assertIn("speed parity", summary["failed"])


if __name__ == "__main__":
    unittest.main()
