from __future__ import annotations

import argparse
import json

from randopt_lora_lab.subspace_audit import CandidateKey
from randopt_lora_lab.subspace_propose import candidate_pool, run


def test_candidate_pool_emits_antithetic_pairs():
    pool = candidate_pool(family="isotropic", population=6, sigma_values=[0.01], seed=7, antithetic=True)
    assert len(pool) == 6
    assert pool[0].seed == pool[1].seed
    assert pool[0].sign == 1
    assert pool[1].sign == -1


def test_subspace_proposal_writes_candidate_file(tmp_path):
    prior = tmp_path / "prior"
    prior.mkdir()
    with (prior / "candidate_summary.jsonl").open("w") as f:
        for seed in range(1, 17):
            cand = CandidateKey("isotropic", seed, 0.01, 1)
            f.write(json.dumps({"candidate": cand.key, "exact_mean": float(seed % 4) / 10.0}) + "\n")

    out = tmp_path / "proposal"
    summary = run(
        argparse.Namespace(
            prior_runs=str(prior),
            out=str(out),
            prior_family="isotropic",
            family="isotropic",
            pool=32,
            keep=8,
            sigma=0.01,
            sigma_values="",
            antithetic=True,
            score_mode="power_energy",
            feature_scale="unit",
            mean_weight=0.25,
            sketch_dim=32,
            components=4,
            power_iter=1,
            max_cap_hit=1.0,
            max_malformed=1.0,
            seed=123,
        )
    )

    rows = [json.loads(line) for line in (out / "candidates.jsonl").read_text().splitlines()]
    assert summary["keep"] == 8
    assert len(rows) == 8
    assert all("candidate" in row for row in rows)
