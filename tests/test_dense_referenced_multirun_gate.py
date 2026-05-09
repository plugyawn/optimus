import json
import tempfile
import unittest
from pathlib import Path

from randopt_lora_lab.dense_referenced_multirun_gate import aggregate, load_run


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def write_run(
    root: Path,
    *,
    prompt_variants: list[str] | None = None,
    search_quality_pass: bool = True,
    shortlist_pass: bool = True,
    full_speedup: float = 5.0,
    strict_delta: float = 0.0,
    malformed: float = 0.0,
    cap_hit: float = 0.0,
) -> None:
    prompt_variants = prompt_variants or ["default", "reordered"]
    write_json(
        root / "shortlist_dense_confirmation" / "summary.json",
        {
            "zero_dense_regret_k": 4,
            "dense_best_recovered_k": 4,
            "dense_best_score": 0.1,
            "gate": {"pass": shortlist_pass, "failed": [] if shortlist_pass else ["dense_regret_threshold"]},
        },
    )
    write_json(
        root / "search_quality_confirmation" / "summary.json",
        {
            "gate": {"pass": search_quality_pass, "failed": [] if search_quality_pass else ["strict_holdout_quality_at_speed"]},
            "rows": [
                {
                    "k": 4,
                    "confirmed_strict_exact": 0.1 + strict_delta,
                    "dense_strict_exact_at_k": 0.1,
                    "delta_vs_dense_best_strict": strict_delta,
                    "full_speedup": full_speedup,
                    "eval_only_speedup": full_speedup + 2.0,
                    "passes_quality": search_quality_pass,
                    "passes_speed": full_speedup >= 1.0,
                }
            ],
        },
    )
    write_json(root / "score_sanity" / "summary.json", {"pass": True, "failed": []})
    write_json(root / "family_state_provenance_audit" / "summary.json", {"pass": True, "failed": []})
    write_json(root / "replay_manifest" / "summary.json", {"artifact_complete": True, "missing_required": []})
    write_json(
        root / "vllm" / "summary.json",
        {
            "candidate_sec": 2.0,
            "prompt_variants": prompt_variants,
            "screen_selection_prompt_variants": prompt_variants,
            "require_all_prompt_variants_valid": True,
            "base_screen_by_prompt": {
                variant: {
                    "cap_hit_mean": cap_hit,
                    "malformed_mean": malformed,
                    "answer_closed_mean": 1.0,
                }
                for variant in prompt_variants
            },
            "base_holdout_by_prompt": {
                variant: {
                    "cap_hit_mean": cap_hit,
                    "malformed_mean": malformed,
                    "answer_closed_mean": 1.0,
                }
                for variant in prompt_variants
            },
        },
    )
    write_json(root / "dense" / "validity" / "summary.json", {"pass": True})
    write_json(root / "confirmed" / "validity" / "summary.json", {"pass": True})


class DenseReferencedMultiRunGateTests(unittest.TestCase):
    def test_load_run_extracts_corrected_confirmation_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            write_run(root, full_speedup=6.0, strict_delta=0.02)

            row = load_run(
                root,
                max_confirm_k=4,
                max_base_cap_hit=0.05,
                max_base_malformed=0.05,
                min_base_answer_closed=1.0,
            )

            self.assertTrue(row["shortlist_dense_pass"])
            self.assertTrue(row["search_quality_pass"])
            self.assertEqual(row["quality_k"], 4)
            self.assertEqual(row["full_speedup"], 6.0)
            self.assertEqual(row["prompt_variant_count"], 2)
            self.assertTrue(row["prompt_health_pass"])

    def test_aggregate_passes_repeated_dense_referenced_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_run(root / "run1", full_speedup=6.0, strict_delta=0.01)
            write_run(root / "run2", full_speedup=8.0, strict_delta=0.03)

            summary = aggregate([root / "run1", root / "run2"], min_runs=2, min_prompt_variants=2)

            self.assertTrue(summary["pass"])
            self.assertEqual(summary["failed"], [])
            self.assertEqual(summary["aggregate"]["search_quality_pass_count"], 2)
            self.assertEqual(summary["aggregate"]["shortlist_dense_pass_count"], 2)
            self.assertEqual(summary["aggregate"]["min_full_speedup"], 6.0)

    def test_aggregate_rejects_prompt_brittle_or_quality_failed_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_run(root / "run1", prompt_variants=["default"])
            write_run(root / "run2", search_quality_pass=False, malformed=0.2)

            summary = aggregate([root / "run1", root / "run2"], min_runs=2, min_prompt_variants=2)

            self.assertFalse(summary["pass"])
            self.assertIn("all_search_quality_pass", summary["failed"])
            self.assertIn("prompt_robust_selection", summary["failed"])


if __name__ == "__main__":
    unittest.main()
