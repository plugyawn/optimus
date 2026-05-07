from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev

import numpy as np

from .compare_backends import spearman


@dataclass(frozen=True)
class CandidateKey:
    family: str
    seed: int
    sigma: float
    sign: int = 1

    @property
    def key(self) -> str:
        return f"{self.family}:seed{self.seed}:s{self.sigma:g}:sign{self.sign}"


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def parse_candidate_key(key: str) -> CandidateKey:
    parts = key.split(":")
    if len(parts) != 4:
        raise ValueError(f"invalid candidate key: {key}")
    return CandidateKey(
        parts[0],
        int(parts[1].removeprefix("seed")),
        float(parts[2].removeprefix("s")),
        int(parts[3].removeprefix("sign")),
    )


@dataclass(frozen=True)
class ScoredCandidate:
    source: str
    candidate: str
    score: float
    family: str
    seed: int
    sigma: float
    sign: int
    cap_hit_mean: float | None = None
    malformed_mean: float | None = None


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def expand_paths(patterns: str) -> list[Path]:
    out: list[Path] = []
    for item in (x.strip() for x in patterns.split(",") if x.strip()):
        matches = glob.glob(item)
        out.extend(Path(x) for x in (matches or [item]))
    return out


def candidate_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return [
        candidate
        for candidate in [
            path / "candidate_summary.jsonl",
            path / "stage_candidate_summary.jsonl",
        ]
        if candidate.exists()
    ]


def row_score(row: dict) -> float | None:
    for key in ("exact_mean", "stage_exact_mean", "score"):
        if key in row and row[key] is not None:
            return float(row[key])
    return None


def load_scored_candidates(
    paths: list[Path],
    *,
    max_cap_hit: float,
    max_malformed: float,
) -> list[ScoredCandidate]:
    out: list[ScoredCandidate] = []
    seen: set[tuple[str, str]] = set()
    for root in paths:
        for file in candidate_files(root):
            for row in read_jsonl(file):
                key = row.get("candidate")
                if not key or key == "base":
                    continue
                score = row_score(row)
                if score is None:
                    continue
                cap_hit = row.get("cap_hit_mean")
                malformed = row.get("malformed_mean")
                if cap_hit is not None and float(cap_hit) > max_cap_hit:
                    continue
                if malformed is not None and float(malformed) > max_malformed:
                    continue
                try:
                    cand = parse_candidate_key(str(key))
                except Exception:
                    continue
                dedupe_key = (str(file), cand.key)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                out.append(
                    ScoredCandidate(
                        source=str(file),
                        candidate=cand.key,
                        score=score,
                        family=cand.family,
                        seed=cand.seed,
                        sigma=cand.sigma,
                        sign=cand.sign,
                        cap_hit_mean=None if cap_hit is None else float(cap_hit),
                        malformed_mean=None if malformed is None else float(malformed),
                    )
                )
    return out


def sketch_seed(candidate: CandidateKey) -> int:
    text = f"{candidate.family}:seed{candidate.seed}:subspace_sketch"
    return stable_int(text) % (2**63 - 1)


def candidate_sketch(candidate: CandidateKey, sketch_dim: int, *, feature_scale: str = "sigma") -> np.ndarray:
    if sketch_dim <= 0:
        raise ValueError("sketch_dim must be positive")
    rng = np.random.default_rng(sketch_seed(candidate))
    x = rng.standard_normal(sketch_dim).astype(np.float64)
    norm = np.linalg.norm(x)
    if norm > 0.0:
        x /= norm
    if feature_scale == "sigma":
        scale = float(candidate.sigma)
    elif feature_scale == "unit":
        scale = 1.0
    else:
        raise ValueError(f"unknown feature_scale: {feature_scale}")
    return float(candidate.sign) * scale * x


def feature_matrix(rows: list[ScoredCandidate], sketch_dim: int, *, feature_scale: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.stack(
        [
            candidate_sketch(CandidateKey(row.family, row.seed, row.sigma, row.sign), sketch_dim, feature_scale=feature_scale)
            for row in rows
        ],
        axis=0,
    )
    y = np.asarray([row.score for row in rows], dtype=np.float64)
    return x, y


def standardize_by_train(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean_x = np.mean(x_train, axis=0, keepdims=True)
    col_scale = np.std(x_train, axis=0)
    col_scale[col_scale < 1e-12] = 1.0
    return (x_train - mean_x) / col_scale[None, :], (x_test - mean_x) / col_scale[None, :]


def ridge_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, ridge: float) -> np.ndarray:
    yc = y_train - float(np.mean(y_train))
    gram = x_train.T @ x_train
    gram += float(ridge) * np.eye(gram.shape[0], dtype=np.float64)
    beta = np.linalg.solve(gram, x_train.T @ yc)
    return x_test @ beta + float(np.mean(y_train))


def mean_direction_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    yc = y_train - float(np.mean(y_train))
    beta = x_train.T @ yc / max(1, x_train.shape[0])
    return x_test @ beta + float(np.mean(y_train))


def power_components(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    components: int,
    n_iter: int,
    seed: int,
) -> np.ndarray:
    k = max(1, min(int(components), x_train.shape[1], x_train.shape[0]))
    weights = np.maximum(y_train - float(np.mean(y_train)), 0.0)
    if not np.any(weights > 0.0):
        weights = np.abs(y_train - float(np.mean(y_train)))
    if not np.any(weights > 0.0):
        weights = np.ones_like(y_train)
    a = x_train * np.sqrt(weights[:, None])
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((a.shape[1], min(a.shape[1], k + 8)))
    q = a.T @ (a @ omega)
    for _ in range(max(0, int(n_iter))):
        q = a.T @ (a @ q)
    q, _ = np.linalg.qr(q)
    small = a @ q
    _, _, vt = np.linalg.svd(small, full_matrices=False)
    comps = vt[:k] @ q.T
    norms = np.linalg.norm(comps, axis=1)
    norms[norms < 1e-12] = 1.0
    return comps / norms[:, None]


def power_energy_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    components: int,
    n_iter: int,
    seed: int,
) -> np.ndarray:
    comps = power_components(x_train, y_train, components=components, n_iter=n_iter, seed=seed)
    projected = x_test @ comps.T
    return np.sum(projected * projected, axis=1)


def topk_metrics(y_true: np.ndarray, y_pred: np.ndarray, top_k: int) -> dict:
    n = len(y_true)
    k = max(1, min(int(top_k), n))
    pred_order = np.argsort(-y_pred)[:k]
    true_order = np.argsort(-y_true)[:k]
    actual_best = float(np.max(y_true))
    selected_best = float(np.max(y_true[pred_order]))
    selected_mean = float(np.mean(y_true[pred_order]))
    baseline_mean = float(np.mean(y_true))
    return {
        "spearman": spearman([float(x) for x in y_pred], [float(x) for x in y_true]),
        "top_k": k,
        "topk_overlap": int(len(set(pred_order.tolist()) & set(true_order.tolist()))),
        "selected_best": selected_best,
        "actual_best": actual_best,
        "regret": actual_best - selected_best,
        "selected_mean": selected_mean,
        "mean_lift": selected_mean - baseline_mean,
    }


def summarize(values: list[float | None]) -> dict:
    nums = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    if not nums:
        return {"mean": None, "std": None}
    return {"mean": mean(nums), "std": pstdev(nums) if len(nums) > 1 else 0.0}


def split_indices(n: int, train_frac: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    order = rng.permutation(n)
    train_n = max(2, min(n - 2, int(round(n * train_frac))))
    return order[:train_n], order[train_n:]


def split_group_indices(labels: list[str], train_frac: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    unique = sorted(set(labels))
    if len(unique) < 2:
        return split_indices(len(labels), train_frac, rng)
    order = rng.permutation(len(unique))
    train_group_n = max(1, min(len(unique) - 1, int(round(len(unique) * train_frac))))
    train_groups = {unique[int(i)] for i in order[:train_group_n]}
    train_idx = np.asarray([idx for idx, label in enumerate(labels) if label in train_groups], dtype=np.int64)
    test_idx = np.asarray([idx for idx, label in enumerate(labels) if label not in train_groups], dtype=np.int64)
    if len(train_idx) < 2 or len(test_idx) < 2:
        return split_indices(len(labels), train_frac, rng)
    return rng.permutation(train_idx), rng.permutation(test_idx)


def make_split(
    n: int,
    train_frac: float,
    rng: np.random.Generator,
    *,
    split_mode: str,
    labels: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if split_mode == "source":
        if labels is None:
            raise ValueError("source split requires labels")
        return split_group_indices(labels, train_frac, rng)
    if split_mode != "row":
        raise ValueError(f"unknown split_mode: {split_mode}")
    return split_indices(n, train_frac, rng)


def run_candidate_audit(
    rows: list[ScoredCandidate],
    *,
    sketch_dim: int,
    components: int,
    splits: int,
    train_frac: float,
    top_k: int,
    ridge: float,
    power_iter: int,
    seed: int,
    split_mode: str = "row",
    feature_scale: str = "sigma",
) -> list[dict]:
    if len(rows) < 8:
        return []
    x, y = feature_matrix(rows, sketch_dim, feature_scale=feature_scale)
    labels = [row.source for row in rows]
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    for split in range(splits):
        train_idx, test_idx = make_split(len(rows), train_frac, rng, split_mode=split_mode, labels=labels)
        x_train, x_test = standardize_by_train(x[train_idx], x[test_idx])
        y_train = y[train_idx]
        y_test = y[test_idx]
        perm_y_train = rng.permutation(y_train)
        predictors = {
            "mean_direction": mean_direction_predict(x_train, y_train, x_test),
            "ridge": ridge_predict(x_train, y_train, x_test, ridge),
            f"power_energy_k{components}": power_energy_predict(
                x_train,
                y_train,
                x_test,
                components=components,
                n_iter=power_iter,
                seed=seed + split,
            ),
            "perm_mean_direction": mean_direction_predict(x_train, perm_y_train, x_test),
            "perm_ridge": ridge_predict(x_train, perm_y_train, x_test, ridge),
            f"perm_power_energy_k{components}": power_energy_predict(
                x_train,
                perm_y_train,
                x_test,
                components=components,
                n_iter=power_iter,
                seed=seed + 10_000 + split,
            ),
        }
        for name, pred in predictors.items():
            records.append({"split": split, "algorithm": name, **topk_metrics(y_test, pred, top_k)})
    return records


def pair_key(row: ScoredCandidate) -> tuple[str, str, int, float]:
    return (row.source, row.family, row.seed, row.sigma)


def antithetic_pairs(rows: list[ScoredCandidate]) -> list[tuple[ScoredCandidate, ScoredCandidate]]:
    by_key: dict[tuple[str, str, int, float], dict[int, ScoredCandidate]] = {}
    for row in rows:
        by_key.setdefault(pair_key(row), {})[row.sign] = row
    pairs = []
    for signs in by_key.values():
        if 1 in signs and -1 in signs:
            pairs.append((signs[1], signs[-1]))
    return pairs


def run_antithetic_audit(
    pairs: list[tuple[ScoredCandidate, ScoredCandidate]],
    *,
    sketch_dim: int,
    splits: int,
    train_frac: float,
    seed: int,
    split_mode: str = "row",
) -> list[dict]:
    if len(pairs) < 8:
        return []
    eps = np.stack(
        [
            candidate_sketch(CandidateKey(pos.family, pos.seed, pos.sigma, 1), sketch_dim, feature_scale="unit")
            for pos, _neg in pairs
        ],
        axis=0,
    )
    diff = np.asarray([pos.score - neg.score for pos, neg in pairs], dtype=np.float64)
    labels = [pos.source for pos, _neg in pairs]
    rng = np.random.default_rng(seed)
    records = []
    for split in range(splits):
        train_idx, test_idx = make_split(len(pairs), train_frac, rng, split_mode=split_mode, labels=labels)
        beta = eps[train_idx].T @ diff[train_idx] / max(1, len(train_idx))
        pred = eps[test_idx] @ beta
        perm_diff = rng.permutation(diff[train_idx])
        perm_beta = eps[train_idx].T @ perm_diff / max(1, len(train_idx))
        perm_pred = eps[test_idx] @ perm_beta
        for name, values in [("antithetic_mean_gradient", pred), ("perm_antithetic_mean_gradient", perm_pred)]:
            actual = diff[test_idx]
            sign_hits = [
                float(np.sign(p) == np.sign(a))
                for p, a in zip(values, actual)
                if p != 0.0 and a != 0.0
            ]
            chosen_scores = []
            random_scores = []
            oracle_scores = []
            for local_idx, prediction in zip(test_idx, values):
                pos, neg = pairs[int(local_idx)]
                chosen_scores.append(pos.score if prediction >= 0.0 else neg.score)
                random_scores.append(0.5 * (pos.score + neg.score))
                oracle_scores.append(max(pos.score, neg.score))
            records.append(
                {
                    "split": split,
                    "algorithm": name,
                    "pair_spearman": spearman([float(x) for x in values], [float(x) for x in actual]),
                    "sign_accuracy": None if not sign_hits else mean(sign_hits),
                    "chosen_mean": mean(chosen_scores),
                    "random_sign_mean": mean(random_scores),
                    "oracle_sign_mean": mean(oracle_scores),
                    "chosen_lift": mean(chosen_scores) - mean(random_scores),
                    "oracle_gap": mean(oracle_scores) - mean(chosen_scores),
                }
            )
    return records


def aggregate_records(records: list[dict], group_key: str) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for row in records:
        groups.setdefault(str(row[group_key]), []).append(row)
    out = []
    for name, rows in sorted(groups.items()):
        keys = sorted({key for row in rows for key in row if key not in {"split", group_key, "top_k"}})
        item = {group_key: name, "splits": len(rows)}
        if "top_k" in rows[0]:
            item["top_k"] = rows[0]["top_k"]
        for key in keys:
            summary = summarize([row.get(key) for row in rows])
            item[f"{key}_mean"] = summary["mean"]
            item[f"{key}_std"] = summary["std"]
        out.append(item)
    return out


def markdown_table(rows: list[dict], cols: list[str]) -> str:
    if not rows:
        return "_No rows._\n"
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in rows:
        vals = []
        for col in cols:
            val = row.get(col)
            if isinstance(val, float):
                vals.append("null" if not math.isfinite(val) else f"{val:.6g}")
            elif val is None:
                vals.append("null")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def write_report(out: Path, summary: dict) -> None:
    candidate_cols = [
        "algorithm",
        "spearman_mean",
        "topk_overlap_mean",
        "regret_mean",
        "mean_lift_mean",
        "selected_best_mean",
    ]
    antithetic_cols = [
        "algorithm",
        "pair_spearman_mean",
        "sign_accuracy_mean",
        "chosen_lift_mean",
        "oracle_gap_mean",
    ]
    body = [
        "# Subspace Audit",
        "",
        f"- Rows loaded: `{summary['rows_loaded']}`",
        f"- Antithetic pairs: `{summary['antithetic_pairs']}`",
        f"- Sketch dim: `{summary['sketch_dim']}`",
        f"- Splits: `{summary['splits']}`",
        f"- Split mode: `{summary['split_mode']}`",
        f"- Feature scale: `{summary['feature_scale']}`",
        "",
        "## Candidate Prediction",
        "",
        markdown_table(summary["candidate_aggregates"], candidate_cols),
        "## Antithetic Direction Test",
        "",
        markdown_table(summary["antithetic_aggregates"], antithetic_cols),
        "",
    ]
    (out / "report.md").write_text("\n".join(body))


def run(args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = expand_paths(args.runs)
    rows = load_scored_candidates(paths, max_cap_hit=args.max_cap_hit, max_malformed=args.max_malformed)
    if args.family:
        families = {x.strip() for x in args.family.split(",") if x.strip()}
        rows = [row for row in rows if row.family in families]
    if args.max_rows and len(rows) > args.max_rows:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(rows), size=args.max_rows, replace=False)
        rows = [rows[int(i)] for i in idx]
    top_k = args.top_k if args.top_k > 0 else max(1, min(32, len(rows) // 16))
    candidate_records = run_candidate_audit(
        rows,
        sketch_dim=args.sketch_dim,
        components=args.components,
        splits=args.splits,
        train_frac=args.train_frac,
        top_k=top_k,
        ridge=args.ridge,
        power_iter=args.power_iter,
        seed=args.seed,
        split_mode=args.split_mode,
        feature_scale=args.feature_scale,
    )
    pairs = antithetic_pairs(rows)
    antithetic_records = run_antithetic_audit(
        pairs,
        sketch_dim=args.sketch_dim,
        splits=args.splits,
        train_frac=args.train_frac,
        seed=args.seed + 12345,
        split_mode=args.split_mode,
    )
    summary = {
        "kind": "subspace_audit",
        "runs": [str(path) for path in paths],
        "rows_loaded": len(rows),
        "families": sorted({row.family for row in rows}),
        "antithetic_pairs": len(pairs),
        "sketch_dim": args.sketch_dim,
        "components": args.components,
        "power_iter": args.power_iter,
        "splits": args.splits,
        "split_mode": args.split_mode,
        "feature_scale": args.feature_scale,
        "train_frac": args.train_frac,
        "top_k": top_k,
        "max_cap_hit": args.max_cap_hit,
        "max_malformed": args.max_malformed,
        "candidate_aggregates": aggregate_records(candidate_records, "algorithm"),
        "antithetic_aggregates": aggregate_records(antithetic_records, "algorithm"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (out / "candidate_split_metrics.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in candidate_records)
    )
    (out / "antithetic_split_metrics.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in antithetic_records)
    )
    write_report(out, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Offline audit for learned/power subspace signal in scored RandOpt candidates.")
    p.add_argument("--runs", required=True, help="Comma-separated run dirs/files/globs containing candidate_summary.jsonl.")
    p.add_argument("--out", required=True)
    p.add_argument("--family", default="", help="Optional comma-separated family filter.")
    p.add_argument("--sketch-dim", type=int, default=256)
    p.add_argument("--components", type=int, default=16)
    p.add_argument("--power-iter", type=int, default=2)
    p.add_argument("--splits", type=int, default=20)
    p.add_argument("--split-mode", choices=["row", "source"], default="row")
    p.add_argument("--feature-scale", choices=["sigma", "unit"], default="sigma")
    p.add_argument("--train-frac", type=float, default=0.5)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--max-cap-hit", type=float, default=0.25)
    p.add_argument("--max-malformed", type=float, default=0.25)
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--seed", type=int, default=1234)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
