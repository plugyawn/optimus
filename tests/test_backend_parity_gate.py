import json
import tempfile
import unittest
from pathlib import Path

from optimus.evaluation.backend_parity import candidate_join_key, main, resolve_adapter_model_path


SUMMARY = {
    "kind": "search",
    "model": "Qwen/Qwen3-4B",
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
    def test_candidate_join_key_matches_legacy_and_method_qualified_keys(self):
        legacy = "isotropic:seed123:s0.0075:sign-1"
        qualified = "lora:isotropic:seed123:s0.0075:sign-1:r8:tq_proj,v_proj"

        self.assertEqual(candidate_join_key(legacy), candidate_join_key(qualified))

    def test_resolve_adapter_model_path_falls_back_to_local_run_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "vllm"
            adapter = candidate / "adapters" / "00000_remote_name"
            adapter.mkdir(parents=True)
            (adapter / "adapter_model.safetensors").write_text("placeholder")

            resolved = resolve_adapter_model_path(
                candidate,
                {"path": "/root/old-project/results/run/vllm/adapters/00000_remote_name"},
            )

            self.assertEqual(resolved, adapter / "adapter_model.safetensors")

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
            self.assertTrue((out / "output_diff_summary.json").exists())

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

    def test_gate_fails_bad_output_diff_even_when_ranking_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trusted = make_run(root, "trusted", [0.3, 0.2, 0.1])
            candidate = make_run(root, "candidate", [0.3, 0.2, 0.1])
            output_diff = root / "output_diff.json"
            write_json(
                output_diff,
                {
                    "exact_disagreement_rate": 0.25,
                    "max_abs_exact_delta_by_candidate": 0.25,
                    "max_abs_cap_hit_delta_by_candidate": 0.0,
                    "max_abs_malformed_delta_by_candidate": 0.0,
                    "answer_equal_rate": 1.0,
                },
            )
            rc = main(
                [
                    "--trusted",
                    str(trusted),
                    "--candidate",
                    str(candidate),
                    "--out",
                    str(root / "gate"),
                    "--allow-missing-adapters",
                    "--output-diff-summary",
                    str(output_diff),
                    "--top8-gate",
                    "3",
                ]
            )
            self.assertEqual(rc, 2)
            summary = json.loads((root / "gate" / "summary.json").read_text())
            self.assertFalse(summary["pass_output_diff"])


if __name__ == "__main__":
    unittest.main()
