from __future__ import annotations

import argparse
import json
from pathlib import Path

from .compare_backends import read_jsonl
from .confirmation_economics import score


def candidate_key(row: dict) -> str:
    return str(row["candidate"])


def sorted_rows(rows: list[dict], score_col: str) -> list[dict]:
    return sorted(rows, key=lambda row: (score(row, score_col), candidate_key(row)), reverse=True)


def write_shortlist(run_dir: Path, out: Path, *, k: int, score_col: str = "exact_mean") -> dict:
    rows = read_jsonl(run_dir / "candidate_summary.jsonl")
    if not rows:
        raise FileNotFoundError(f"missing candidate rows in {run_dir}")
    selected = sorted_rows(rows, score_col)[: min(k, len(rows))]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for rank, row in enumerate(selected, start=1):
            f.write(
                json.dumps(
                    {
                        "rank": rank,
                        "candidate": candidate_key(row),
                        "score": score(row, score_col),
                        "score_col": score_col,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    return {
        "kind": "shortlist_from_run",
        "run_dir": str(run_dir),
        "out": str(out),
        "score_col": score_col,
        "requested_k": k,
        "written": len(selected),
        "top_candidate": candidate_key(selected[0]) if selected else None,
        "top_score": score(selected[0], score_col) if selected else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write top-K candidate keys from a run's candidate_summary.jsonl.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--score-col", default="exact_mean")
    args = parser.parse_args(argv)

    summary = write_shortlist(args.run, args.out, k=args.k, score_col=args.score_col)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
