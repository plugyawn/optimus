import json
from pathlib import Path

from randopt_lora_lab.score_sanity_audit import analyze


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def run_summary() -> dict:
    return {
        "family": "activation_spectral_lora_c2",
        "targets": ["q_proj"],
        "population": 4,
        "screen_prompts": 64,
        "holdout_prompts": 128,
        "base_screen_exact": 0.05,
        "base_holdout_exact": 0.08,
        "base_screen_by_prompt": {
            "default": {"exact_mean": 0.05, "cap_hit_mean": 0.0, "malformed_mean": 0.0, "answer_closed_mean": 1.0},
            "reordered": {"exact_mean": 0.06, "cap_hit_mean": 0.0, "malformed_mean": 0.02, "answer_closed_mean": 1.0},
        },
    }


def candidate(candidate: str, exact: float, *, score: float | None = None, cap: float = 0.0, malformed: float = 0.0) -> dict:
    row = {
        "candidate": candidate,
        "exact_mean": exact,
        "cap_hit_mean": cap,
        "malformed_mean": malformed,
        "answer_closed_mean": 1.0,
    }
    if score is not None:
        row["selection_score"] = score
    return row


def test_score_sanity_passes_on_clean_topk_even_if_tail_is_bad(tmp_path: Path):
    root = tmp_path / "run"
    write_json(root / "summary.json", run_summary())
    write_jsonl(
        root / "candidate_summary.jsonl",
        [
            candidate("good-a", 0.09, score=1.0, cap=0.0, malformed=0.0),
            candidate("good-b", 0.08, score=0.9, cap=0.02, malformed=0.03),
            candidate("bad-tail", 0.0, score=-1.0, cap=1.0, malformed=1.0),
        ],
    )

    summary = analyze(root, top_k=2)

    assert summary["pass"] is True
    assert summary["failed"] == []
    run = summary["runs"][0]
    assert run["metrics"]["all_max_cap_hit"] == 1.0
    assert run["metrics"]["topk_max_cap_hit"] == 0.02


def test_score_sanity_fails_when_top_candidate_is_cap_hit(tmp_path: Path):
    root = tmp_path / "run"
    write_json(root / "summary.json", run_summary())
    write_jsonl(
        root / "candidate_summary.jsonl",
        [
            candidate("bad-selected", 0.09, score=1.0, cap=0.5, malformed=0.0),
            candidate("clean", 0.08, score=0.5, cap=0.0, malformed=0.0),
        ],
    )

    summary = analyze(root, top_k=1)

    assert summary["pass"] is False
    assert "run:topk_cap_hit_below_threshold" in summary["failed"]


def test_score_sanity_fails_when_best_topk_does_not_clear_base(tmp_path: Path):
    root = tmp_path / "run"
    payload = run_summary()
    payload["base_screen_exact"] = 0.2
    write_json(root / "summary.json", payload)
    write_jsonl(root / "candidate_summary.jsonl", [candidate("weak", 0.1)])

    summary = analyze(root, top_k=1)

    assert summary["pass"] is False
    assert "run:topk_best_exact_clears_base" in summary["failed"]


def test_score_sanity_discovers_dense_vllm_confirmed_children(tmp_path: Path):
    for child in ["dense", "vllm", "confirmed"]:
        root = tmp_path / "panel" / child
        write_json(root / "summary.json", run_summary())
        write_jsonl(root / "candidate_summary.jsonl", [candidate(f"{child}-good", 0.1)])

    summary = analyze(tmp_path / "panel")

    assert summary["pass"] is True
    assert [run["name"] for run in summary["runs"]] == ["dense", "vllm", "confirmed"]
