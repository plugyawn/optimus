from __future__ import annotations

import json
from pathlib import Path

import torch

from optimus.search.adaptive import candidate_score_rows, col_scale_from_rows, top_basis


def test_candidate_score_rows_reads_summary_and_deduplicates_best_candidate(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    key = "lora:factor_gaussian_lora:seed1:s0.01:sign1:r8:tq_proj,v_proj"
    (run / "candidate_summary.jsonl").write_text(
        json.dumps({"candidate": key, "exact_mean": 0.1}) + "\n"
        + json.dumps({"candidate": "bad-key", "exact_mean": 1.0}) + "\n"
    )
    (run / "summary.json").write_text(
        json.dumps({"top_screen": [{"candidate": key, "exact_mean": 0.25}]}) + "\n"
    )

    rows = candidate_score_rows([run], top_k=8, min_score=0.0)

    assert len(rows) == 1
    assert rows[0]["candidate"] == key
    assert rows[0]["score_for_basis"] == 0.25
    assert rows[0]["source"].endswith("summary.json")


def test_top_basis_returns_centered_svd_rows():
    basis = top_basis([torch.tensor([[1.0, 0.0], [0.0, 1.0]])], basis_rank=1)

    assert basis is not None
    assert tuple(basis.shape) == (1, 2)
    assert torch.isclose(basis.norm(), torch.tensor(1.0), atol=1e-6)


def test_col_scale_from_rows_clamps_and_keeps_shape():
    rows = [torch.tensor([[1.0, 3.0, 9.0], [1.0, 3.0, 9.0]])]
    scale = col_scale_from_rows(rows, strength=1.0, clamp=(0.5, 2.0))

    assert scale is not None
    assert tuple(scale.shape) == (3,)
    assert float(scale.min()) >= 0.5
    assert float(scale.max()) <= 2.0
