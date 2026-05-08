import json
from pathlib import Path

import pytest

from randopt_lora_lab.shortlist_dense_confirmation import analyze, gate
from randopt_lora_lab.shortlist_from_run import write_shortlist


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def write_run(
    root: Path,
    family: str,
    scores: list[float],
    *,
    seeds: list[int] | None = None,
    candidate_sec: float = 1.0,
    elapsed: float = 1.0,
) -> None:
    seeds = seeds or list(range(1, len(scores) + 1))
    write_json(root / "summary.json", {"population": len(scores), "candidate_sec": candidate_sec})
    write_jsonl(
        root / "candidate_summary.jsonl",
        [
            {
                "candidate": f"{family}:seed{seed}:s0.001:sign1",
                "exact_mean": score,
                "elapsed_s": elapsed,
            }
            for seed, score in zip(seeds, scores)
        ],
    )


def test_write_shortlist_sorts_by_score_then_candidate_key(tmp_path: Path):
    run = tmp_path / "proposal"
    write_run(run, "sparse_low_rank_lora_d0p125", [0.1, 0.3, 0.2])

    summary = write_shortlist(run, tmp_path / "shortlist.jsonl", k=2)

    rows = [json.loads(line) for line in (tmp_path / "shortlist.jsonl").read_text().splitlines()]
    assert summary["written"] == 2
    assert [row["candidate"] for row in rows] == [
        "sparse_low_rank_lora_d0p125:seed2:s0.001:sign1",
        "sparse_low_rank_lora_d0p125:seed3:s0.001:sign1",
    ]


def test_shortlist_dense_confirmation_uses_full_dense_best_not_shortlist_best(tmp_path: Path):
    dense = tmp_path / "dense"
    proposal = tmp_path / "proposal"
    confirmed = tmp_path / "confirmed"
    write_run(dense, "dense_gaussian", [0.9, 0.3, 0.2, 0.1], candidate_sec=1.0)
    write_run(proposal, "sparse_low_rank_lora_d0p125", [0.1, 0.8, 0.7, 0.0], candidate_sec=10.0)
    write_run(confirmed, "sparse_low_rank_lora_d0p125", [0.8, 0.7], seeds=[2, 3], candidate_sec=2.0)

    rows, summary = analyze(dense, confirmed, proposal, ks=[1, 2, 4])

    assert summary["dense_best_spec"] == "seed1:s0.001:sign1"
    assert summary["zero_dense_regret_k"] is None
    assert summary["dense_best_recovered_k"] == 4
    assert rows[1]["confirmed_spec"] == "seed2:s0.001:sign1"
    assert rows[1]["dense_regret_vs_best"] == pytest.approx(0.6)
    assert not gate(rows, summary, max_confirm_k=2)["pass"]


def test_shortlist_dense_confirmation_passes_when_shortlist_confirms_dense_best(tmp_path: Path):
    dense = tmp_path / "dense"
    proposal = tmp_path / "proposal"
    confirmed = tmp_path / "confirmed"
    write_run(dense, "dense_gaussian", [0.9, 0.3], candidate_sec=1.0)
    write_run(proposal, "sparse_low_rank_lora_d0p125", [0.8, 0.1], candidate_sec=10.0)
    write_run(confirmed, "sparse_low_rank_lora_d0p125", [0.8], candidate_sec=2.0)

    rows, summary = analyze(dense, confirmed, proposal, ks=[1])

    assert summary["zero_dense_regret_k"] == 1
    assert gate(rows, summary, max_confirm_k=1, min_full_without_dense_load_speedup=0.1)["pass"]
