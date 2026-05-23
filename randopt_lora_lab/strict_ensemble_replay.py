from __future__ import annotations

import argparse
import json
from pathlib import Path

from optimus.tasks.countdown import CountdownExample
from .experiments import majority_vote_evaluation, write_jsonl


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def examples_from_rows(rows: list[dict]) -> list[CountdownExample]:
    by_id = {}
    for row in rows:
        example_id = int(row["example_id"])
        by_id.setdefault(
            example_id,
            CountdownExample(
                example_id,
                tuple(int(x) for x in row["numbers"]),
                int(row["target"]),
            ),
        )
    return [by_id[key] for key in sorted(by_id)]


def candidate_order(summary: dict) -> list[str]:
    top = summary.get("top_screen") or []
    return [str(row["candidate"]) for row in top]


def replay(run_dir: Path, *, out: Path | None = None) -> dict:
    summary = json.loads((run_dir / "summary.json").read_text())
    holdout_rows = read_jsonl(run_dir / "holdout_per_prompt.jsonl")
    examples = examples_from_rows(holdout_rows)
    order = candidate_order(summary)
    k_values = [int(k) for k in summary.get("ensemble_ks", [])]
    if not k_values:
        raise ValueError(f"{run_dir} has no ensemble_ks in summary.json")
    if not order:
        raise ValueError(f"{run_dir} has no top_screen candidate order in summary.json")
    strict_summary, strict_per_prompt = majority_vote_evaluation(
        order,
        holdout_rows,
        examples,
        k_values,
        strict_rows=True,
    )
    numeric_summary, numeric_per_prompt = majority_vote_evaluation(order, holdout_rows, examples, k_values)
    payload = {
        "kind": "strict_ensemble_replay",
        "run_dir": str(run_dir),
        "candidate_order": order,
        "ensemble_ks": k_values,
        "numeric_ensemble_holdout": numeric_summary,
        "strict_ensemble_holdout": strict_summary,
        "best_numeric_ensemble_holdout_exact": max((row["exact_mean"] for row in numeric_summary), default=None),
        "best_strict_ensemble_holdout_exact": max((row["exact_mean"] for row in strict_summary), default=None),
    }
    out_dir = out or run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "strict_ensemble_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    per_prompt_rows = [dict(row, vote_filter="numeric") for row in numeric_per_prompt]
    per_prompt_rows.extend(dict(row, vote_filter="strict_numeric") for row in strict_per_prompt)
    per_prompt_path = out_dir / "strict_ensemble_per_prompt.jsonl"
    if per_prompt_path.exists():
        per_prompt_path.unlink()
    write_jsonl(per_prompt_path, per_prompt_rows)
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Replay numeric and strict ensemble votes from saved holdout rows.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    payload = replay(Path(args.run), out=Path(args.out) if args.out else None)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
