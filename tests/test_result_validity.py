import json
import tempfile
import unittest
from pathlib import Path

from randopt_lora_lab.result_validity import run_validity_audit


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def row(mode: str, candidate: str, example_id: int, numbers: list[int], target: int, text: str, **extra) -> dict:
    payload = {
        "mode": mode,
        "candidate": candidate,
        "example_id": example_id,
        "numbers": numbers,
        "target": target,
        "text": text,
        "output_tokens": 6,
        "cap_hit": 0.0,
        "answer_closed": 1.0,
        "exact": 1.0,
        "malformed": False,
        "missing_answer": False,
        "multiple_answers": False,
        "trailing_text": False,
        "answer_count": 1,
    }
    payload.update(extra)
    return payload


class ResultValidityTests(unittest.TestCase):
    def make_valid_run(self, root: Path) -> Path:
        run = root / "run"
        write_json(
            run / "summary.json",
            {
                "screen_prompts": 1,
                "holdout_prompts": 1,
                "screen_unique_prompts": 1,
                "holdout_unique_prompts": 1,
                "screen_unique_semantic_prompts": 1,
                "holdout_unique_semantic_prompts": 1,
                "screen_holdout_overlap": 0,
                "candidate_score_metric": "exact_answer",
                "ensemble_vote_metric": "valid_numeric_majority_vote",
                "ensemble_ks": [1],
                "top_holdout": [{"cap_hit_mean": 0.0, "malformed_mean": 0.0}],
            },
        )
        write_jsonl(
            run / "per_prompt.jsonl",
            [
                row("base_screen", "base", 1, [1, 2, 3], 6, "<answer>1+2+3</answer>"),
                row("screen", "isotropic:seed1:s0.01:sign1", 1, [1, 2, 3], 6, "<answer>1+2+3</answer>"),
            ],
        )
        write_jsonl(
            run / "holdout_per_prompt.jsonl",
            [
                row("base_holdout", "base", 2, [2, 3, 4], 9, "<answer>2+3+4</answer>"),
                row("holdout", "isotropic:seed1:s0.01:sign1", 2, [2, 3, 4], 9, "<answer>2+3+4</answer>"),
            ],
        )
        write_jsonl(
            run / "ensemble_per_prompt.jsonl",
            [
                {
                    "k": 1,
                    "example_id": 2,
                    "numbers": [2, 3, 4],
                    "target": 9,
                    "final_vote": "9",
                    "exact": 1.0,
                    "valid_vote_count": 1,
                    "missing_vote_count": 0,
                    "vote_counts": {"9": 1},
                    "reject_counts": {},
                }
            ],
        )
        return run

    def test_valid_run_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_validity_audit(self.make_valid_run(Path(tmp)))

        self.assertTrue(summary["pass"])
        self.assertEqual(summary["failed"], [])

    def test_fails_repeated_holdout_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make_valid_run(Path(tmp))
            rows = [
                row("base_holdout", "base", 2, [2, 3, 4], 9, "<answer>2+3+4</answer>"),
                row("base_holdout", "base", 3, [4, 3, 2], 9, "<answer>4+3+2</answer>"),
                row("holdout", "isotropic:seed1:s0.01:sign1", 2, [2, 3, 4], 9, "<answer>2+3+4</answer>"),
            ]
            write_jsonl(run / "holdout_per_prompt.jsonl", rows)
            summary = run_validity_audit(run)

        self.assertFalse(summary["pass"])
        self.assertIn("holdout_base_semantics_unique[default]", summary["failed"])

    def test_fails_stale_parser_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make_valid_run(Path(tmp))
            write_jsonl(
                run / "holdout_per_prompt.jsonl",
                [
                    row("base_holdout", "base", 2, [2, 3, 4], 9, "<answer>2+3+4</answer> trailing"),
                    row("holdout", "isotropic:seed1:s0.01:sign1", 2, [2, 3, 4], 9, "<answer>2+3+4</answer>"),
                ],
            )
            summary = run_validity_audit(run)

        self.assertFalse(summary["pass"])
        self.assertIn("stored_rows_match_current_strict_parser", summary["failed"])

    def test_fails_high_selected_cap_hit(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make_valid_run(Path(tmp))
            write_jsonl(
                run / "holdout_per_prompt.jsonl",
                [
                    row("base_holdout", "base", 2, [2, 3, 4], 9, "<answer>2+3+4</answer>"),
                    row(
                        "holdout",
                        "isotropic:seed1:s0.01:sign1",
                        2,
                        [2, 3, 4],
                        9,
                        "<answer>2+3+4</answer>",
                        cap_hit=1.0,
                    ),
                ],
            )
            summary = run_validity_audit(run, max_selected_cap_hit=0.1)

        self.assertFalse(summary["pass"])
        self.assertIn("selected_candidate_cap_hit_below_threshold", summary["failed"])

    def test_allows_same_examples_across_prompt_variants(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make_valid_run(Path(tmp))
            screen_rows = []
            holdout_rows = []
            for variant in ["default", "reordered"]:
                screen_rows.append(
                    row(
                        "base_screen",
                        "base",
                        1,
                        [1, 2, 3],
                        6,
                        "<answer>1+2+3</answer>",
                        prompt_variant=variant,
                    )
                )
                holdout_rows.extend(
                    [
                        row(
                            "base_holdout",
                            "base",
                            2,
                            [2, 3, 4],
                            9,
                            "<answer>2+3+4</answer>",
                            prompt_variant=variant,
                        ),
                        row(
                            "holdout",
                            "isotropic:seed1:s0.01:sign1",
                            2,
                            [2, 3, 4],
                            9,
                            "<answer>2+3+4</answer>",
                            prompt_variant=variant,
                        ),
                    ]
                )
            write_jsonl(run / "per_prompt.jsonl", screen_rows)
            write_jsonl(run / "holdout_per_prompt.jsonl", holdout_rows)

            summary = run_validity_audit(run)

        self.assertTrue(summary["pass"], summary["failed"])


if __name__ == "__main__":
    unittest.main()
