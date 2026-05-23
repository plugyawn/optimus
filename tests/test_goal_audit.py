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
            local_baseline_audit=None,
            parity_report=None,
            backend_gate=None,
            confirmation_gate=None,
            dense_confirmation_gate=None,
            search_quality_confirmation=None,
            family_state_provenance=None,
            multirun_gate=None,
            prompt_robustness=None,
            drift_report=None,
            eval_validity=None,
            score_sanity=None,
            adapter_run=None,
        )

        summary = run_goal_audit(args)

        self.assertFalse(summary["pass"])
        self.assertIn("upstream dense baseline validity", summary["failed"])
        self.assertIn("quality parity", summary["failed"])
        self.assertIn("accelerated evaluation route", summary["failed"])
        self.assertIn("adapter identity provenance", summary["failed"])
        self.assertIn("multi-run prompt-robust confirmation", summary["failed"])
        self.assertIn("drift parity", summary["failed"])
        self.assertIn("eval validity", summary["failed"])
        self.assertIn("score sanity", summary["failed"])
        action_by_requirement = {row["requirement"]: row for row in summary["next_actions"]}
        self.assertEqual(summary["next_actions"][0]["requirement"], "accelerated evaluation route")
        self.assertEqual(summary["next_actions"][1]["requirement"], "adapter identity provenance")
        self.assertEqual(
            action_by_requirement["accelerated evaluation route"]["command"],
            "MODE=confirm scripts/run_qproj_c2_corrected_confirmation.sh",
        )
        self.assertLess(
            action_by_requirement["accelerated evaluation route"]["priority"],
            action_by_requirement["upstream dense baseline validity"]["priority"],
        )
        self.assertEqual(
            action_by_requirement["adapter identity provenance"]["command"],
            "MODE=confirm scripts/run_qproj_c2_corrected_confirmation.sh",
        )

    def test_full_evidence_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local_baseline = root / "local_baseline" / "summary.json"
            parity = root / "parity" / "summary.json"
            backend = root / "backend" / "summary.json"
            confirmation = root / "confirmation" / "summary.json"
            provenance = root / "provenance" / "summary.json"
            multirun = root / "multirun" / "summary.json"
            prompt = root / "prompt" / "summary.json"
            drift = root / "drift" / "summary.json"
            eval_validity = root / "eval_validity" / "summary.json"
            score_sanity = root / "score_sanity" / "summary.json"
            adapter = root / "adapter"
            write_json(local_baseline, {"pass": True})
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
            write_json(provenance, {"pass": True, "failed": [], "runs": [{"root": "run", "pass": True}]})
            write_json(multirun, {"pass": True, "failed": [], "aggregate": {"runs": 2}})
            write_json(prompt, {"gate": {"pass": True, "valid_prompt_variants": 2, "passing_prompt_variants": 2, "min_valid_prompts": 2}})
            write_json(drift, {"pass": True})
            write_json(eval_validity, {"pass": True})
            write_json(score_sanity, {"pass": True})
            write_json(adapter / "summary.json", {"adapters_kept": True})
            args = Namespace(
                local_baseline_audit=local_baseline,
                parity_report=parity,
                backend_gate=backend,
                confirmation_gate=confirmation,
                dense_confirmation_gate=None,
                search_quality_confirmation=None,
                family_state_provenance=provenance,
                multirun_gate=multirun,
                prompt_robustness=prompt,
                drift_report=drift,
                eval_validity=eval_validity,
                score_sanity=score_sanity,
                adapter_run=adapter,
            )

            summary = run_goal_audit(args)

            self.assertTrue(summary["pass"])
            self.assertEqual(summary["failed"], [])
            self.assertEqual(summary["next_actions"], [])

    def test_upstream_full_scale_audit_can_satisfy_official_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            upstream = root / "upstream" / "summary.json"
            write_json(
                upstream,
                {
                    "pass": True,
                    "smoke_pass": True,
                    "upstream_scale_pass": True,
                    "failed": [],
                    "summary": {"population_size": 5000, "train_samples": 200, "max_tokens": 1024},
                },
            )
            args = Namespace(
                local_baseline_audit=None,
                upstream_baseline_audit=upstream,
                parity_report=None,
                backend_gate=None,
                confirmation_gate=None,
                dense_confirmation_gate=None,
                search_quality_confirmation=None,
                family_state_provenance=None,
                multirun_gate=None,
                prompt_robustness=None,
                drift_report=None,
                eval_validity=None,
                score_sanity=None,
                adapter_run=None,
            )

            summary = run_goal_audit(args)
            by_requirement = {row["requirement"]: row for row in summary["checks"]}

            self.assertTrue(by_requirement["upstream dense baseline validity"]["passed"])
            self.assertEqual(
                by_requirement["upstream dense baseline validity"]["detail"]["routes"],
                ["upstream_full_scale"],
            )

    def test_upstream_smoke_audit_does_not_satisfy_official_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            upstream = root / "upstream" / "summary.json"
            write_json(
                upstream,
                {
                    "pass": True,
                    "smoke_pass": True,
                    "upstream_scale_pass": False,
                    "failed": [],
                    "summary": {"population_size": 32, "train_samples": 200, "max_tokens": 1024},
                },
            )
            args = Namespace(
                local_baseline_audit=None,
                upstream_baseline_audit=upstream,
                parity_report=None,
                backend_gate=None,
                confirmation_gate=None,
                dense_confirmation_gate=None,
                search_quality_confirmation=None,
                family_state_provenance=None,
                multirun_gate=None,
                prompt_robustness=None,
                drift_report=None,
                eval_validity=None,
                score_sanity=None,
                adapter_run=None,
            )

            summary = run_goal_audit(args)
            by_requirement = {row["requirement"]: row for row in summary["checks"]}
            detail = by_requirement["upstream dense baseline validity"]["detail"]

            self.assertFalse(by_requirement["upstream dense baseline validity"]["passed"])
            self.assertTrue(detail["upstream_countdown"]["smoke_pass"])
            self.assertFalse(detail["upstream_countdown"]["upstream_scale_pass"])

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
                local_baseline_audit=None,
                parity_report=parity,
                backend_gate=None,
                confirmation_gate=None,
                dense_confirmation_gate=None,
                search_quality_confirmation=None,
                family_state_provenance=None,
                multirun_gate=None,
                prompt_robustness=None,
                drift_report=None,
                eval_validity=None,
                score_sanity=None,
                adapter_run=None,
            )

            summary = run_goal_audit(args)

            self.assertIn("quality parity", summary["failed"])
            self.assertIn("stability parity", summary["failed"])
            self.assertIn("speed parity", summary["failed"])

    def test_nested_parity_report_uses_named_arm(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parity = root / "parity" / "summary.json"
            write_json(
                parity,
                {
                    "pass": False,
                    "comparisons": {
                        "lora": {
                            "pass": True,
                            "spearman": 0.9,
                            "topk_overlap": 8,
                            "selected_regret": 0.0,
                            "ensemble_holdout_delta_lora_minus_dense": 0.0,
                            "speed_ratio_lora_over_dense": 1.2,
                            "gates": {
                                "ensemble_quality": True,
                                "spearman": True,
                                "topk_overlap": True,
                                "selected_regret": True,
                                "speed": True,
                            },
                        },
                        "control": {
                            "pass": False,
                            "gates": {},
                        },
                    },
                },
            )
            args = Namespace(
                local_baseline_audit=None,
                parity_report=parity,
                parity_arm="lora",
                backend_gate=None,
                confirmation_gate=None,
                dense_confirmation_gate=None,
                search_quality_confirmation=None,
                family_state_provenance=None,
                multirun_gate=None,
                prompt_robustness=None,
                drift_report=None,
                eval_validity=None,
                score_sanity=None,
                adapter_run=None,
            )

            summary = run_goal_audit(args)
            by_requirement = {row["requirement"]: row for row in summary["checks"]}

            self.assertTrue(by_requirement["quality parity"]["passed"])
            self.assertTrue(by_requirement["stability parity"]["passed"])
            self.assertTrue(by_requirement["speed parity"]["passed"])

    def test_same_family_confirmation_does_not_satisfy_accelerated_route(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backend = root / "backend" / "summary.json"
            confirmation = root / "confirmation" / "summary.json"
            provenance = root / "provenance" / "summary.json"
            multirun = root / "multirun" / "summary.json"
            write_json(backend, {"pass": False, "failed": ["ranking"]})
            write_json(
                confirmation,
                {
                    "best_recovered_k": 1,
                    "zero_regret_k": 1,
                    "gate": {"pass": True, "failed": [], "thresholds": {"max_confirm_k": 16}},
                },
            )
            write_json(multirun, {"pass": False, "failed": ["all_quality_parity_pass"]})
            write_json(provenance, {"pass": False, "failed": ["results/stale_run"]})
            args = Namespace(
                local_baseline_audit=None,
                parity_report=None,
                backend_gate=backend,
                confirmation_gate=confirmation,
                dense_confirmation_gate=None,
                search_quality_confirmation=None,
                family_state_provenance=provenance,
                multirun_gate=multirun,
                prompt_robustness=None,
                drift_report=None,
                eval_validity=None,
                score_sanity=None,
                adapter_run=None,
            )

            summary = run_goal_audit(args)
            by_requirement = {row["requirement"]: row for row in summary["checks"]}

            self.assertFalse(by_requirement["accelerated evaluation route"]["passed"])
            self.assertTrue(by_requirement["accelerated evaluation route"]["detail"]["same_family_confirmation_pass"])
            self.assertFalse(by_requirement["adapter identity provenance"]["passed"])
            self.assertFalse(by_requirement["multi-run prompt-robust confirmation"]["passed"])
            self.assertIn("accelerated evaluation route", summary["failed"])
            self.assertIn("adapter identity provenance", summary["failed"])

    def test_dense_referenced_two_stage_route_can_satisfy_acceleration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backend = root / "backend" / "summary.json"
            dense_confirmation = root / "dense_confirmation" / "summary.json"
            search_quality = root / "search_quality" / "summary.json"
            write_json(backend, {"pass": False, "failed": ["ranking"]})
            write_json(
                dense_confirmation,
                {
                    "kind": "shortlist_dense_confirmation",
                    "zero_dense_regret_k": 4,
                    "dense_best_recovered_k": 4,
                    "gate": {"pass": True, "failed": [], "thresholds": {"max_confirm_k": 8}},
                },
            )
            write_json(
                search_quality,
                {
                    "kind": "search_quality_confirmation",
                    "gate": {"pass": True, "failed": [], "thresholds": {"min_full_speedup": 1.0}},
                },
            )
            args = Namespace(
                local_baseline_audit=None,
                parity_report=None,
                backend_gate=backend,
                confirmation_gate=None,
                dense_confirmation_gate=dense_confirmation,
                search_quality_confirmation=search_quality,
                family_state_provenance=None,
                multirun_gate=None,
                prompt_robustness=None,
                drift_report=None,
                eval_validity=None,
                score_sanity=None,
                adapter_run=None,
            )

            summary = run_goal_audit(args)
            by_requirement = {row["requirement"]: row for row in summary["checks"]}

            self.assertTrue(by_requirement["accelerated evaluation route"]["passed"])
            self.assertTrue(by_requirement["accelerated evaluation route"]["detail"]["dense_referenced_two_stage_pass"])
            self.assertEqual(by_requirement["accelerated evaluation route"]["detail"]["routes"], ["dense_referenced_two_stage"])

    def test_dense_referenced_multirun_can_satisfy_parity_prompt_and_multirun(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dense_multirun = root / "dense_multirun" / "summary.json"
            write_json(
                dense_multirun,
                {
                    "pass": True,
                    "failed": [],
                    "aggregate": {
                        "runs": 2,
                        "search_quality_pass_count": 2,
                        "shortlist_dense_pass_count": 2,
                        "prompt_robust_count": 2,
                        "min_full_speedup": 5.9,
                    },
                    "thresholds": {"min_runs": 2, "min_prompt_variants": 2},
                },
            )
            args = Namespace(
                local_baseline_audit=None,
                parity_report=None,
                backend_gate=None,
                confirmation_gate=None,
                dense_confirmation_gate=None,
                search_quality_confirmation=None,
                family_state_provenance=None,
                multirun_gate=None,
                dense_referenced_multirun_gate=dense_multirun,
                prompt_robustness=None,
                drift_report=None,
                eval_validity=None,
                score_sanity=None,
                adapter_run=None,
            )

            summary = run_goal_audit(args)
            by_requirement = {row["requirement"]: row for row in summary["checks"]}

            self.assertTrue(by_requirement["quality parity"]["passed"])
            self.assertTrue(by_requirement["stability parity"]["passed"])
            self.assertTrue(by_requirement["speed parity"]["passed"])
            self.assertTrue(by_requirement["multi-run prompt-robust confirmation"]["passed"])
            self.assertTrue(by_requirement["prompt robustness"]["passed"])
            self.assertIn("dense_referenced_multirun", by_requirement["quality parity"]["detail"]["routes"])
            self.assertIn("dense_referenced_multirun", by_requirement["prompt robustness"]["detail"]["routes"])

    def test_score_sanity_failure_blocks_goal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            score_sanity = root / "score_sanity" / "summary.json"
            write_json(score_sanity, {"pass": False, "failed": ["vllm:base_screen_prompt_health"]})
            args = Namespace(
                local_baseline_audit=None,
                parity_report=None,
                backend_gate=None,
                confirmation_gate=None,
                dense_confirmation_gate=None,
                search_quality_confirmation=None,
                family_state_provenance=None,
                multirun_gate=None,
                prompt_robustness=None,
                drift_report=None,
                eval_validity=None,
                score_sanity=score_sanity,
                adapter_run=None,
            )

            summary = run_goal_audit(args)
            by_requirement = {row["requirement"]: row for row in summary["checks"]}

            self.assertFalse(by_requirement["score sanity"]["passed"])
            self.assertIn("score sanity", summary["failed"])
            self.assertEqual(by_requirement["score sanity"]["detail"]["failed"], ["vllm:base_screen_prompt_health"])


if __name__ == "__main__":
    unittest.main()
