from __future__ import annotations

import numpy as np

from randopt_lora_lab.subspace_audit import (
    CandidateKey,
    ScoredCandidate,
    antithetic_pairs,
    candidate_sketch,
    run_antithetic_audit,
    run_candidate_audit,
)


def scored(candidate: CandidateKey, score: float) -> ScoredCandidate:
    return ScoredCandidate(
        source="synthetic",
        candidate=candidate.key,
        score=score,
        family=candidate.family,
        seed=candidate.seed,
        sigma=candidate.sigma,
        sign=candidate.sign,
    )


def test_candidate_sketch_preserves_antithetic_negation():
    pos = CandidateKey("isotropic", 17, 0.01, 1)
    neg = CandidateKey("isotropic", 17, 0.01, -1)
    assert np.allclose(candidate_sketch(pos, 32), -candidate_sketch(neg, 32))


def test_candidate_audit_beats_permuted_control_on_linear_synthetic_signal():
    dim = 64
    hidden = np.linspace(-1.0, 1.0, dim)
    hidden = hidden / np.linalg.norm(hidden)
    rows = []
    for seed in range(1, 129):
        cand = CandidateKey("isotropic", seed, 0.01, 1)
        signal = float(candidate_sketch(cand, dim) @ hidden)
        rows.append(scored(cand, 0.5 + 20.0 * signal))

    metrics = run_candidate_audit(
        rows,
        sketch_dim=dim,
        components=8,
        splits=5,
        train_frac=0.5,
        top_k=8,
        ridge=1e-2,
        power_iter=1,
        seed=99,
    )
    by_alg = {}
    for row in metrics:
        by_alg.setdefault(row["algorithm"], []).append(row["spearman"])

    mean_direction = np.nanmean(by_alg["mean_direction"])
    perm_direction = np.nanmean(by_alg["perm_mean_direction"])
    assert mean_direction > 0.5
    assert mean_direction > perm_direction + 0.3


def test_antithetic_audit_recovers_synthetic_sign_preference():
    dim = 64
    hidden = np.linspace(1.0, -1.0, dim)
    hidden = hidden / np.linalg.norm(hidden)
    rows = []
    for seed in range(1, 257):
        pos = CandidateKey("isotropic", seed, 0.01, 1)
        neg = CandidateKey("isotropic", seed, 0.01, -1)
        signal = float(candidate_sketch(pos, dim) @ hidden)
        rows.append(scored(pos, 0.5 + 20.0 * signal))
        rows.append(scored(neg, 0.5 - 20.0 * signal))

    pairs = antithetic_pairs(rows)
    metrics = run_antithetic_audit(pairs, sketch_dim=dim, splits=5, train_frac=0.5, seed=123)
    by_alg = {}
    for row in metrics:
        by_alg.setdefault(row["algorithm"], []).append(row["sign_accuracy"])

    real = np.nanmean(by_alg["antithetic_mean_gradient"])
    perm = np.nanmean(by_alg["perm_antithetic_mean_gradient"])
    assert real > 0.7
    assert real > perm + 0.2
