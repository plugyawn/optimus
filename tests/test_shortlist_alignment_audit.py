import json
from pathlib import Path

import pytest

from randopt_lora_lab.shortlist_alignment_audit import analyze


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def cand(family: str, seed: int, score: float, **extra) -> dict:
    row = {
        "candidate": f"{family}:seed{seed}:s0.001:sign1",
        "exact_mean": score,
        "selection_score": score,
        "seed": seed,
        "sigma": 0.001,
        "sign": 1,
    }
    row.update(extra)
    return row


def condition(family: str, seed: int, variant: str, score: float) -> dict:
    row = cand(family, seed, score)
    row["prompt_variant"] = variant
    row["condition_selection_score"] = score
    return row


def prompt_row(family: str, seed: int, example_id: int, exact: float, text: str, *, variant: str | None = None) -> dict:
    row = {
        "candidate": f"{family}:seed{seed}:s0.001:sign1",
        "example_id": example_id,
        "exact": exact,
        "mode": "screen",
        "text": text,
    }
    if variant is not None:
        row["prompt_variant"] = variant
    return row


def test_alignment_audit_separates_dense_recall_and_backend_agreement(tmp_path: Path):
    root = tmp_path / "run"
    write_jsonl(
        root / "dense" / "candidate_summary.jsonl",
        [
            cand("dense_gaussian", 1, 0.9),
            cand("dense_gaussian", 2, 0.4),
            cand("dense_gaussian", 3, 0.1),
        ],
    )
    write_jsonl(
        root / "vllm" / "candidate_summary.jsonl",
        [
            cand("lora", 1, 0.1, selection_score=0.1),
            cand("lora", 2, 0.8, selection_score=0.8),
            cand("lora", 3, 0.7, selection_score=0.7),
        ],
    )
    write_jsonl(
        root / "confirmed" / "candidate_summary.jsonl",
        [
            cand("lora", 2, 0.6),
            cand("lora", 3, 0.2),
        ],
    )
    write_jsonl(
        root / "vllm" / "candidate_condition_summary.jsonl",
        [
            condition("lora", 1, "default", 0.1),
            condition("lora", 1, "reordered", 0.3),
            condition("lora", 2, "default", 0.8),
            condition("lora", 2, "reordered", 0.2),
            condition("lora", 3, "default", 0.7),
            condition("lora", 3, "reordered", 0.1),
        ],
    )
    write_jsonl(
        root / "vllm" / "per_prompt.jsonl",
        [
            prompt_row("lora", 2, 10, 1.0, "a", variant="default"),
            prompt_row("lora", 2, 11, 0.0, "b", variant="default"),
        ],
    )
    write_jsonl(
        root / "confirmed" / "per_prompt.jsonl",
        [
            prompt_row("lora", 2, 10, 1.0, "a"),
            prompt_row("lora", 2, 11, 1.0, "c"),
        ],
    )

    summary = analyze(root, ks=[1, 2])

    recall = summary["dense_vs_proposal"]["dense_recall_by_selection"]
    assert recall["dense_best_spec"] == "seed1:s0.001:sign1"
    assert recall["dense_best_proposal_rank"] == 3
    assert recall["rows"][0]["contains_dense_best"] is False
    assert recall["rows"][1]["dense_topk_overlap"] == 1

    confirmed = summary["confirmed_vs_proposal"]["proposal_exact"]
    assert confirmed["common"] == 2
    assert confirmed["mean_abs_delta"] == pytest.approx(0.35)

    per_prompt = summary["per_prompt_default_vllm_vs_peft"]
    assert per_prompt["common_rows"] == 2
    assert per_prompt["exact_equal_fraction"] == 0.5
    assert per_prompt["text_equal_fraction"] == 0.5
