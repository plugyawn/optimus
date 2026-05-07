from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def run_label(run_dir: Path) -> str:
    return run_dir.name


def condition_rows(run_dir: Path) -> list[dict]:
    summary_path = run_dir / "summary.json"
    args_path = run_dir / "args.json"
    if not summary_path.exists() or not args_path.exists():
        return []
    summary = read_json(summary_path)
    args = read_json(args_path)
    rows = []
    for condition, metrics in sorted(summary.get("conditions", {}).items()):
        hf_cap = float(metrics.get("hf_cap_hit_mean", 0.0))
        vllm_cap = float(metrics.get("vllm_cap_hit_mean", 0.0))
        hf_malformed = float(metrics.get("hf_malformed_mean", 0.0))
        vllm_malformed = float(metrics.get("vllm_malformed_mean", 0.0))
        rows.append(
            {
                "run": run_label(run_dir),
                "condition": condition,
                "n": int(metrics.get("n", 0)),
                "text_equal_rate": float(metrics.get("text_equal_rate", 0.0)),
                "answer_equal_rate": float(metrics.get("answer_equal_rate", 0.0)),
                "exact_equal_rate": float(metrics.get("exact_equal_rate", 0.0)),
                "hf_exact_mean": float(metrics.get("hf_exact_mean", 0.0)),
                "vllm_exact_mean": float(metrics.get("vllm_exact_mean", 0.0)),
                "hf_cap_hit_mean": hf_cap,
                "vllm_cap_hit_mean": vllm_cap,
                "abs_cap_hit_delta": abs(vllm_cap - hf_cap),
                "hf_malformed_mean": hf_malformed,
                "vllm_malformed_mean": vllm_malformed,
                "abs_malformed_delta": abs(vllm_malformed - hf_malformed),
                "mean_abs_output_token_delta": float(metrics.get("mean_abs_output_token_delta", 0.0)),
                "vllm_dtype": args.get("vllm_dtype"),
                "adapter_dtype": args.get("adapter_dtype"),
                "enforce_eager": bool(args.get("enforce_eager")),
                "stop_at_answer": bool(args.get("stop_at_answer")),
                "max_new_tokens": int(args.get("max_new_tokens", summary.get("max_new_tokens", 0))),
            }
        )
    return rows


def rollout_valid(row: dict, *, min_text_equal: float, max_cap_delta: float, max_malformed_delta: float, max_token_delta: float) -> bool:
    return (
        float(row["text_equal_rate"]) >= min_text_equal
        and float(row["abs_cap_hit_delta"]) <= max_cap_delta
        and float(row["abs_malformed_delta"]) <= max_malformed_delta
        and float(row["mean_abs_output_token_delta"]) <= max_token_delta
    )


def summarize(rows: list[dict], *, min_text_equal: float, max_cap_delta: float, max_malformed_delta: float, max_token_delta: float) -> dict:
    for row in rows:
        row["rollout_valid"] = rollout_valid(
            row,
            min_text_equal=min_text_equal,
            max_cap_delta=max_cap_delta,
            max_malformed_delta=max_malformed_delta,
            max_token_delta=max_token_delta,
        )
    by_run = {}
    for run in sorted({row["run"] for row in rows}):
        subset = [row for row in rows if row["run"] == run]
        by_run[run] = {
            "conditions": len(subset),
            "rollout_valid_conditions": sum(1 for row in subset if row["rollout_valid"]),
            "min_text_equal_rate": min((row["text_equal_rate"] for row in subset), default=None),
            "max_abs_cap_hit_delta": max((row["abs_cap_hit_delta"] for row in subset), default=None),
            "max_abs_malformed_delta": max((row["abs_malformed_delta"] for row in subset), default=None),
            "max_mean_abs_output_token_delta": max((row["mean_abs_output_token_delta"] for row in subset), default=None),
            "pass": all(row["rollout_valid"] for row in subset) if subset else False,
        }
    passing_runs = [run for run, metrics in by_run.items() if metrics["pass"]]
    return {
        "kind": "backend_rollout_ablation_report",
        "runs": len(by_run),
        "condition_rows": len(rows),
        "min_text_equal": min_text_equal,
        "max_cap_delta": max_cap_delta,
        "max_malformed_delta": max_malformed_delta,
        "max_token_delta": max_token_delta,
        "passing_runs": passing_runs,
        "by_run": by_run,
        "pass": bool(passing_runs),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def write_report(path: Path, rows: list[dict], summary: dict) -> None:
    run_rows = [
        {"run": run, **metrics}
        for run, metrics in sorted(summary["by_run"].items())
    ]
    lines = [
        "# Backend Rollout Ablation Report",
        "",
        f"Pass: `{str(summary['pass']).lower()}`",
        "",
        "| gate | value |",
        "| --- | ---: |",
        f"| runs | {summary['runs']} |",
        f"| condition rows | {summary['condition_rows']} |",
        f"| min text equality | {summary['min_text_equal']} |",
        f"| max cap-hit delta | {summary['max_cap_delta']} |",
        f"| max malformed delta | {summary['max_malformed_delta']} |",
        f"| max mean token-count delta | {summary['max_token_delta']} |",
        "",
        "## Runs",
        "",
        markdown_table(
            run_rows,
            [
                "run",
                "conditions",
                "rollout_valid_conditions",
                "min_text_equal_rate",
                "max_abs_cap_hit_delta",
                "max_abs_malformed_delta",
                "max_mean_abs_output_token_delta",
                "pass",
            ],
        ),
        "",
        "## Conditions",
        "",
        markdown_table(
            rows,
            [
                "run",
                "condition",
                "text_equal_rate",
                "answer_equal_rate",
                "exact_equal_rate",
                "abs_cap_hit_delta",
                "abs_malformed_delta",
                "mean_abs_output_token_delta",
                "rollout_valid",
            ],
        ),
        "",
    ]
    path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Summarize backend rollout ablation runs.")
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--min-text-equal", type=float, default=0.95)
    p.add_argument("--max-cap-delta", type=float, default=0.05)
    p.add_argument("--max-malformed-delta", type=float, default=0.05)
    p.add_argument("--max-token-delta", type=float, default=1.0)
    args = p.parse_args(argv)

    rows = []
    for run_dir in sorted(path for path in args.root.iterdir() if path.is_dir()):
        rows.extend(condition_rows(run_dir))
    summary = summarize(
        rows,
        min_text_equal=args.min_text_equal,
        max_cap_delta=args.max_cap_delta,
        max_malformed_delta=args.max_malformed_delta,
        max_token_delta=args.max_token_delta,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "conditions.csv", rows)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_report(args.out / "report.md", rows, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
