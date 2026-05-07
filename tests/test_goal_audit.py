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
            confirmation_gate=None,
            prompt_robustness=None,
            drift_report=None,
            eval_validity=None,
            adapter_run=None,
        )

        summary = run_goal_audit(args)

        self.assertFalse(summary["pass"])
        self.assertIn("official full-Gaussian baseline validity", summary["failed"])
        self.assertIn("quality parity", summary["failed"])
        self.assertIn("two-stage accelerated confirmation", summary["failed"])
        self.assertIn("drift parity", summary["failed"])
        self.assertIn("eval validity", summary["failed"])

    def test_full_evidence_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reproduction = root / "repro" / "summary.json"
            parity = root / "parity" / "summary.json"
            backend = root / "backend" / "summary.json"
            confirmation = root / "confirmation" / "summary.json"
            prompt = root / "prompt" / "summary.json"
            drift = root / "drift" / "summary.json"
            eval_validity = root / "eval_validity" / "summary.json"
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
            write_json(confirmation, {"gate": {"pass": True}, "best_recovered_k": 1, "zero_regret_k": 1})
            write_json(prompt, {"gate": {"pass": True, "valid_prompt_variants": 2, "passing_prompt_variants": 2, "min_valid_prompts": 2}})
            write_json(drift, {"pass": True})
            write_json(eval_validity, {"pass": True})
            write_json(adapter / "summary.json", {"adapters_kept": True})
            args = Namespace(
                reproduction_audit=reproduction,
                parity_report=parity,
                backend_gate=backend,
                confirmation_gate=confirmation,
                prompt_robustness=prompt,
                drift_report=drift,
                eval_validity=eval_validity,
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
                confirmation_gate=None,
                prompt_robustness=None,
                drift_report=None,
                eval_validity=None,
                adapter_run=None,
            )

            summary = run_goal_audit(args)

            self.assertIn("quality parity", summary["failed"])
            self.assertIn("stability parity", summary["failed"])
            self.assertIn("speed parity", summary["failed"])

    def test_confirmation_gate_is_separate_from_backend_selector(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backend = root / "backend" / "summary.json"
            confirmation = root / "confirmation" / "summary.json"
            write_json(backend, {"pass": False, "failed": ["ranking"]})
            write_json(
                confirmation,
                {
                    "best_recovered_k": 1,
                    "zero_regret_k": 1,
                    "gate": {"pass": True, "failed": [], "thresholds": {"max_confirm_k": 16}},
                },
            )
            args = Namespace(
                reproduction_audit=None,
                parity_report=None,
                backend_gate=backend,
                confirmation_gate=confirmation,
                prompt_robustness=None,
                drift_report=None,
                eval_validity=None,
                adapter_run=None,
            )

            summary = run_goal_audit(args)
            by_requirement = {row["requirement"]: row for row in summary["checks"]}

            self.assertFalse(by_requirement["trusted accelerated backend selector"]["passed"])
            self.assertTrue(by_requirement["two-stage accelerated confirmation"]["passed"])
            self.assertIn("trusted accelerated backend selector", summary["failed"])
            self.assertNotIn("two-stage accelerated confirmation", summary["failed"])


if __name__ == "__main__":
    unittest.main()
