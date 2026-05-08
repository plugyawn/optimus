import json
import tempfile
import unittest
from pathlib import Path

from randopt_lora_lab.multirun_gate import aggregate, load_run, select_parity_arm


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def write_run(
    root: Path,
    *,
    parity_pass: bool = True,
    confirmation_pass: bool = True,
    prompt_variants: list[str] | None = None,
    selected_regret: float = 0.0,
    spearman: float = 0.9,
    zero_regret_k: int | None = 2,
    full_speedup: float = 2.0,
) -> None:
    prompt_variants = prompt_variants or ["default", "xml"]
    write_json(
        root / "parity" / "summary.json",
        {
            "pass": parity_pass,
            "comparisons": {
                "lora": {
                    "pass": parity_pass,
                    "spearman": spearman,
                    "topk_overlap": 8,
                    "selected_regret": selected_regret,
                    "ensemble_holdout_delta_lora_minus_dense": 0.0,
                    "speed_ratio_lora_over_dense": 1.1,
                    "mutation_s_ratio_lora_over_dense": 0.1,
                    "gates": {
                        "spearman": parity_pass,
                        "topk_overlap": True,
                        "selected_regret": selected_regret <= 0.0,
                        "speed": True,
                        "ensemble_quality": True,
                    },
                }
            },
        },
    )
    write_json(
        root / "confirmation" / "summary.json",
        {
            "zero_regret_k": zero_regret_k,
            "best_recovered_k": zero_regret_k,
            "gate": {"pass": confirmation_pass, "failed": [] if confirmation_pass else ["zero_regret_within_k"]},
            "rows": []
            if zero_regret_k is None
            else [
                {
                    "k": zero_regret_k,
                    "regret_vs_trusted_best": 0.0,
                    "full_without_peft_load_speedup_vs_trusted_full": full_speedup,
                    "eval_only_speedup_vs_trusted_full": full_speedup + 1.0,
                }
            ],
        },
    )
    write_json(
        root / "vllm_spectral" / "summary.json",
        {
            "screen_selection_prompt_variants": prompt_variants,
            "base_screen_exact": 0.1,
            "base_screen_by_prompt": {name: {"cap_hit_mean": 0.0, "malformed_mean": 0.0} for name in prompt_variants},
        },
    )
    for arm in ["dense", "control", "spectral"]:
        write_json(root / arm / "validity" / "summary.json", {"pass": True})


class MultiRunGateTests(unittest.TestCase):
    def test_selects_named_arm_from_nested_parity_report(self):
        summary = {"comparisons": {"lora": {"pass": True, "spearman": 0.9}}}
        self.assertEqual(select_parity_arm(summary, "lora")["spearman"], 0.9)
        self.assertFalse(select_parity_arm(summary, "missing")["pass"])

    def test_load_run_extracts_quality_systems_and_prompt_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            write_run(root)

            row = load_run(root, parity_arm="lora", validity_arms=["dense", "control", "spectral"])

            self.assertTrue(row["parity_pass"])
            self.assertTrue(row["confirmation_pass"])
            self.assertEqual(row["prompt_variant_count"], 2)
            self.assertEqual(row["zero_regret_k"], 2)
            self.assertEqual(row["full_without_load_speedup_at_zero_regret"], 2.0)

    def test_aggregate_passes_only_when_all_strict_gates_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_run(root / "run1")
            write_run(root / "run2")

            summary = aggregate([root / "run1", root / "run2"], min_runs=2, min_prompt_variants=2)

            self.assertTrue(summary["pass"])
            self.assertEqual(summary["failed"], [])
            self.assertEqual(summary["aggregate"]["parity_pass_count"], 2)

    def test_aggregate_fails_for_single_default_prompt_quality_negative(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            write_run(root, parity_pass=False, prompt_variants=["default"], selected_regret=0.01, spearman=0.3)

            summary = aggregate([root], min_runs=2, min_prompt_variants=2)

            self.assertFalse(summary["pass"])
            self.assertIn("min_runs", summary["failed"])
            self.assertIn("all_quality_parity_pass", summary["failed"])
            self.assertIn("prompt_robust_selection", summary["failed"])


if __name__ == "__main__":
    unittest.main()
