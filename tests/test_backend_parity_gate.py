import json
import tempfile
import unittest
from pathlib import Path

from randopt_lora_lab.backend_parity_gate import main


SUMMARY = {
    "kind": "search",
    "model": "Qwen/Qwen2.5-3B-Instruct",
    "family": "factor_gaussian_lora",
    "population": 3,
    "rank": 8,
    "sigma": 0.0075,
    "targets": "q_proj,v_proj",
    "screen_prompts": 4,
    "max_new_tokens": 32,
    "stop_at_answer": True,
    "antithetic": False,
    "screen_holdout_overlap": 0,
}


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def make_run(root: Path, name: str, scores: list[float], *, summary_extra: dict | None = None) -> Path:
    run = root / name
    run.mkdir()
    summary = dict(SUMMARY)
    summary.update(summary_extra or {})
    write_json(run / "summary.json", summary)
    write_jsonl(run / "per_prompt.jsonl", [{"mode": "base_screen", "candidate": "base"}])
    write_jsonl(run / "holdout_per_prompt.jsonl", [{"mode": "base_holdout", "candidate": "base"}])
    rows = [{"candidate": f"c{i}", "exact_mean": score} for i, score in enumerate(scores)]
    write_jsonl(run / "candidate_summary.jsonl", rows)
    return run


class BackendParityGateTests(unittest.TestCase):
    def test_gate_can_pass_with_missing_adapters_explicitly_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trusted = make_run(root, "trusted", [0.3, 0.2, 0.1])
            candidate = make_run(root, "candidate", [0.3, 0.2, 0.1])
            out = root / "gate"
            rc = main(
                [
                    "--trusted",
                    str(trusted),
                    "--candidate",
                    str(candidate),
                    "--out",
                    str(out),
                    "--allow-missing-adapters",
                    "--top8-gate",
                    "3",
                ]
            )
            self.assertEqual(rc, 0)
            summary = json.loads((out / "summary.json").read_text())
            self.assertTrue(summary["pass"])

    def test_gate_fails_missing_adapters_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trusted = make_run(root, "trusted", [0.3, 0.2, 0.1])
            candidate = make_run(root, "candidate", [0.3, 0.2, 0.1])
            rc = main(["--trusted", str(trusted), "--candidate", str(candidate), "--out", str(root / "gate"), "--top8-gate", "3"])
            self.assertEqual(rc, 2)

    def test_gate_fails_protocol_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trusted = make_run(root, "trusted", [0.3, 0.2, 0.1])
            candidate = make_run(root, "candidate", [0.3, 0.2, 0.1], summary_extra={"max_new_tokens": 64})
            rc = main(
                [
                    "--trusted",
                    str(trusted),
                    "--candidate",
                    str(candidate),
                    "--out",
                    str(root / "gate"),
                    "--allow-missing-adapters",
                    "--top8-gate",
                    "3",
                ]
            )
            self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
