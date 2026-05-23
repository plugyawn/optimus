from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def candidate_rows(run_dir: Path) -> list[dict]:
    rows = read_jsonl(run_dir / "candidate_summary.jsonl")
    if not rows:
        rows = read_jsonl(run_dir / "stage_candidate_summary.jsonl")
    if not rows:
        raise FileNotFoundError(f"no candidate summary jsonl found in {run_dir}")
    required = {"candidate", "exact_mean"}
    for idx, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise ValueError(f"{run_dir} candidate row {idx} missing columns: {sorted(missing)}")
    return rows


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for idx in order[i:j]:
            ranks[idx] = avg_rank
        i = j
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx = mean(xs)
    my = mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = sum((x - mx) ** 2 for x in xs) ** 0.5
    den_y = sum((y - my) ** 2 for y in ys) ** 0.5
    if den_x == 0.0 or den_y == 0.0:
        return None
    return num / (den_x * den_y)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    return pearson(average_ranks(xs), average_ranks(ys))


def top_candidates(rows: list[dict], score_col: str, k: int) -> set[str]:
    return {row["candidate"] for row in sorted(rows, key=lambda r: r[score_col], reverse=True)[:k]}


def compare(
    trusted_dir: Path,
    candidate_dir: Path,
    *,
    trusted_name: str,
    candidate_name: str,
    ks: list[int],
    spearman_gate: float,
    top8_gate: int,
) -> tuple[list[dict], dict]:
    trusted_rows = candidate_rows(trusted_dir)
    candidate_backend_rows = candidate_rows(candidate_dir)
    trusted_score = f"{trusted_name}_exact_mean"
    candidate_score = f"{candidate_name}_exact_mean"
    by_candidate = {
        row["candidate"]: {trusted_score: float(row["exact_mean"])}
        for row in trusted_rows
    }
    for row in candidate_backend_rows:
        if row["candidate"] in by_candidate:
            by_candidate[row["candidate"]][candidate_score] = float(row["exact_mean"])
    joined = [
        {"candidate": key, **value}
        for key, value in by_candidate.items()
        if candidate_score in value
    ]
    xs = [row[trusted_score] for row in joined]
    ys = [row[candidate_score] for row in joined]
    trusted_desc_ranks = average_ranks([-x for x in xs])
    candidate_desc_ranks = average_ranks([-y for y in ys])
    for row, trusted_rank, candidate_rank in zip(joined, trusted_desc_ranks, candidate_desc_ranks):
        row["abs_score_delta"] = abs(row[trusted_score] - row[candidate_score])
        row["trusted_rank"] = trusted_rank
        row["candidate_rank"] = candidate_rank
        row["rank_delta"] = abs(trusted_rank - candidate_rank)

    overlaps = {}
    for k in ks:
        k_eff = min(k, len(joined))
        overlaps[f"top{k}_overlap"] = len(
            top_candidates(joined, trusted_score, k_eff) & top_candidates(joined, candidate_score, k_eff)
        )
        overlaps[f"top{k}_possible"] = k_eff

    trusted_best = max(joined, key=lambda r: r[trusted_score]) if joined else {}
    candidate_best = max(joined, key=lambda r: r[candidate_score]) if joined else {}
    selected_regret = None
    if trusted_best and candidate_best:
        selected_regret = float(trusted_best[trusted_score] - candidate_best[trusted_score])
    top8_overlap = overlaps.get("top8_overlap")
    pass_top8 = top8_overlap is not None and top8_overlap >= min(top8_gate, overlaps.get("top8_possible", top8_gate))
    rho = spearman(xs, ys)
    summary = {
        "kind": "backend_parity",
        "trusted_dir": str(trusted_dir),
        "candidate_dir": str(candidate_dir),
        "trusted_name": trusted_name,
        "candidate_name": candidate_name,
        "n_trusted": int(len(trusted_rows)),
        "n_candidate": int(len(candidate_backend_rows)),
        "n_common": int(len(joined)),
        "pearson": pearson(xs, ys),
        "spearman": rho,
        "mean_abs_score_delta": mean([row["abs_score_delta"] for row in joined]) if joined else None,
        "max_abs_score_delta": max([row["abs_score_delta"] for row in joined]) if joined else None,
        "trusted_best_candidate": trusted_best.get("candidate"),
        "trusted_best_score": trusted_best.get(trusted_score),
        "candidate_best_candidate": candidate_best.get("candidate"),
        "candidate_best_trusted_score": candidate_best.get(trusted_score),
        "candidate_best_score": candidate_best.get(candidate_score),
        "selected_regret_vs_trusted": selected_regret,
        "spearman_gate": spearman_gate,
        "top8_gate": top8_gate,
        "pass_spearman_gate": rho is not None and rho >= spearman_gate,
        "pass_top8_gate": pass_top8,
        **overlaps,
    }
    summary["pass"] = bool(summary["pass_spearman_gate"] and summary["pass_top8_gate"])
    return sorted(joined, key=lambda r: r[candidate_score], reverse=True), summary


def parse_ks(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def markdown_table(rows: list[dict], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict]) -> None:
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, joined: list[dict], summary: dict) -> None:
    status = "PASS" if summary["pass"] else "FAIL"
    lines = [
        "# Backend Parity Report",
        "",
        f"Status: **{status}**",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| common candidates | {summary['n_common']} |",
        f"| Spearman | {summary['spearman']} |",
        f"| Pearson | {summary['pearson']} |",
        f"| mean abs score delta | {summary['mean_abs_score_delta']} |",
        f"| selected regret vs trusted | {summary['selected_regret_vs_trusted']} |",
    ]
    for key, value in summary.items():
        if key.startswith("top") and key.endswith("_overlap"):
            possible = summary.get(key.replace("_overlap", "_possible"))
            lines.append(f"| {key} | {value}/{possible} |")
    lines.extend(
        [
            "",
            "## Winners",
            "",
            f"- Trusted best: `{summary['trusted_best_candidate']}` at `{summary['trusted_best_score']}`.",
            f"- Candidate-backend selected: `{summary['candidate_best_candidate']}`; trusted score `{summary['candidate_best_trusted_score']}`, candidate-backend score `{summary['candidate_best_score']}`.",
            "",
            "## Top Candidate-Backend Rows",
            "",
            markdown_table(joined[:16], list(joined[0].keys()) if joined else ["candidate"]),
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Compare matched candidate rankings across two backends.")
    p.add_argument("--trusted", required=True, type=Path)
    p.add_argument("--candidate", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--trusted-name", default="trusted")
    p.add_argument("--candidate-name", default="candidate")
    p.add_argument("--ks", default="4,8,16")
    p.add_argument("--spearman-gate", type=float, default=0.85)
    p.add_argument("--top8-gate", type=int, default=6)
    args = p.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    joined, summary = compare(
        args.trusted,
        args.candidate,
        trusted_name=args.trusted_name,
        candidate_name=args.candidate_name,
        ks=parse_ks(args.ks),
        spearman_gate=args.spearman_gate,
        top8_gate=args.top8_gate,
    )
    write_csv(args.out / "joined.csv", joined)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_report(args.out / "report.md", joined, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
