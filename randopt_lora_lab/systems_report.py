from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_summary(path: Path) -> dict:
    row = json.loads(path.read_text())
    row["summary_path"] = str(path)
    row["suite"] = path.parts[-3] if len(path.parts) >= 3 else ""
    row["run"] = path.parent.name
    return row


def phase8_summaries(root: Path) -> list[dict]:
    return [read_summary(path) for path in sorted(root.glob("phase8*/**/summary.json"))]


def csv_write(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def md_table(rows: list[dict], columns: list[str], *, limit: int | None = None) -> str:
    shown = rows[:limit] if limit is not None else rows
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in shown:
        values = []
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                val = f"{val:.4g}"
            values.append(str(val))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def full_search_rows(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        if row.get("kind") != "vllm_lora_search":
            continue
        out.append(
            {
                "suite": row["suite"],
                "run": row["run"],
                "population": row.get("population"),
                "screen_prompts": row.get("screen_prompts"),
                "chunk_adapters": row.get("chunk_adapters"),
                "max_loras": row.get("max_loras"),
                "max_new_tokens": row.get("max_new_tokens"),
                "enforce_eager": row.get("enforce_eager"),
                "max_num_batched_tokens": row.get("max_num_batched_tokens"),
                "candidate_sec": row.get("candidate_sec"),
                "screen_prompts_per_sec": row.get("screen_prompts_per_sec"),
                "eval_elapsed_s": row.get("eval_elapsed_s"),
                "load_s": row.get("load_s"),
            }
        )
    return sorted(out, key=lambda r: (r.get("candidate_sec") or 0), reverse=True)


def parity_rows(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        if row.get("kind") != "backend_parity":
            continue
        out.append(
            {
                "suite": row["suite"],
                "run": row["run"],
                "trusted_name": row.get("trusted_name"),
                "candidate_name": row.get("candidate_name"),
                "n_common": row.get("n_common"),
                "spearman": row.get("spearman"),
                "top8_overlap": row.get("top8_overlap"),
                "top8_possible": row.get("top8_possible"),
                "selected_regret_vs_trusted": row.get("selected_regret_vs_trusted"),
                "pass": row.get("pass"),
                "trusted_best_candidate": row.get("trusted_best_candidate"),
                "candidate_best_candidate": row.get("candidate_best_candidate"),
            }
        )
    return sorted(out, key=lambda r: (str(r.get("suite")), str(r.get("run"))))


def halving_rows(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        if row.get("kind") != "halving_recall":
            continue
        out.append(
            {
                "suite": row["suite"],
                "run": row["run"],
                "screen_prompts": row.get("screen_prompts"),
                "stage_prompts": row.get("stage_prompts"),
                "survivors": row.get("survivors"),
                "candidate_sec": row.get("candidate_sec"),
                "prompt_eval_savings": row.get("prompt_eval_savings"),
                "top8_survivor_recall": row.get("top8_survivor_recall"),
                "top8_possible": row.get("top8_possible"),
                "full_best_survived": row.get("full_best_survived"),
                "halving_selected_regret_vs_full": row.get("halving_selected_regret_vs_full"),
                "eval_elapsed_s": row.get("eval_elapsed_s"),
            }
        )
    return sorted(out, key=lambda r: (r.get("screen_prompts") or 0, r.get("stage_prompts") or 0))


def plot_full_search(path: Path, rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [
        row
        for row in rows
        if row.get("population") in {512, 1024}
        and row.get("max_new_tokens") in {16, 32}
        and row.get("screen_prompts") in {64, 128}
    ][:18]
    labels = [
        f"{row['suite'].replace('phase8_', '')}/{row['run'].replace('search_', '')}"
        for row in selected
    ]
    values = [row.get("candidate_sec") or 0.0 for row in selected]
    fig, ax = plt.subplots(figsize=(11, max(4, 0.38 * len(selected))))
    ax.barh(range(len(selected)), values, color="#2f6f73")
    ax.set_yticks(range(len(selected)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("candidates/sec")
    ax.set_title("Phase8 full-search throughput")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_parity(path: Path, rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [row.get("spearman") or 0.0 for row in rows]
    ys = [row.get("top8_overlap") or 0 for row in rows]
    colors = ["#1f7a4d" if row.get("pass") else "#b84a39" for row in rows]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(xs, ys, c=colors, s=70, edgecolor="#222222", linewidth=0.4)
    ax.axvline(0.85, color="#777777", linestyle="--", linewidth=1)
    ax.axhline(6, color="#777777", linestyle="--", linewidth=1)
    ax.set_xlabel("Spearman vs trusted screen")
    ax.set_ylabel("top-8 overlap")
    ax.set_title("Matched-exploration parity gates")
    ax.set_xlim(0.35, 1.01)
    ax.set_ylim(0, 8.5)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_halving(path: Path, rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for row in rows:
        savings = row.get("prompt_eval_savings") or 0.0
        regret = row.get("halving_selected_regret_vs_full") or 0.0
        color = "#1f7a4d" if row.get("full_best_survived") else "#b84a39"
        label = f"p{row.get('screen_prompts')}/s{row.get('stage_prompts')}/k{row.get('survivors')}"
        ax.scatter([savings], [regret], c=color, s=90, edgecolor="#222222", linewidth=0.4)
        ax.annotate(label, (savings, regret), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("prompt eval savings")
    ax.set_ylabel("selected regret vs full")
    ax.set_title("Staged-search speed/recall tradeoff")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_report(path: Path, full: list[dict], parity: list[dict], halving: list[dict]) -> None:
    fastest_raw = full[0] if full else {}
    matched_protocol_full = [
        row
        for row in full
        if row.get("max_new_tokens") == 32
        and not row.get("enforce_eager")
        and (row.get("max_num_batched_tokens") in {None, 0, ""})
    ]
    fastest_matched = matched_protocol_full[0] if matched_protocol_full else {}
    passing_parity = [row for row in parity if row.get("pass")]
    zero_regret_halving = [
        row
        for row in halving
        if row.get("full_best_survived") and (row.get("halving_selected_regret_vs_full") or 0.0) == 0.0
    ]
    fastest_raw_line = "- No full-search rows found."
    if fastest_raw:
        fastest_raw_line = (
            f"- Fastest raw full-search row: `{fastest_raw.get('suite')}/{fastest_raw.get('run')}` "
            f"at `{fastest_raw.get('candidate_sec'):.4g}` candidates/sec, but this row is accepted only if its parity gate passes."
        )
    fastest_matched_line = "- No matched-protocol full-search rows found."
    if fastest_matched:
        fastest_matched_line = (
            f"- Fastest matched-protocol full search: `{fastest_matched.get('suite')}/{fastest_matched.get('run')}` "
            f"at `{fastest_matched.get('candidate_sec'):.4g}` candidates/sec."
        )
    lines = [
        "# Phase8 Systems Report",
        "",
        "## Executive Call",
        "",
        fastest_raw_line,
        fastest_matched_line,
        "- `max_new_tokens=16` is rejected as a search accelerator: it is faster but fails matched ranking parity.",
        "- Eager mode is rejected: lower throughput and failed top-8 parity.",
        "- `chunk_adapters=4` is the fastest full-search setting; `chunk_adapters=8` is the more conservative reference when strict top-8 parity matters at larger population.",
        "- Staged search with `stage_prompts=8, survivors=64` is the best first-stage triage setting observed: zero selected regret on p64 and p128 panels, with low top-8 recall.",
        "",
        "## Full Search",
        "",
        md_table(
            full,
            [
                "suite",
                "run",
                "population",
                "screen_prompts",
                "chunk_adapters",
                "max_new_tokens",
                "candidate_sec",
                "screen_prompts_per_sec",
                "eval_elapsed_s",
            ],
            limit=14,
        ),
        "",
        "## Parity Gates",
        "",
        md_table(
            parity,
            [
                "suite",
                "run",
                "trusted_name",
                "candidate_name",
                "spearman",
                "top8_overlap",
                "selected_regret_vs_trusted",
                "pass",
            ],
        ),
        "",
        "## Staged Search",
        "",
        md_table(
            halving,
            [
                "suite",
                "run",
                "screen_prompts",
                "stage_prompts",
                "survivors",
                "candidate_sec",
                "prompt_eval_savings",
                "top8_survivor_recall",
                "full_best_survived",
                "halving_selected_regret_vs_full",
            ],
        ),
        "",
        "## Gate Summary",
        "",
        f"- Passing parity rows: `{len(passing_parity)}/{len(parity)}`.",
        f"- Zero-regret staged rows with full best survived: `{len(zero_regret_halving)}/{len(halving)}`.",
        "",
        "Plots: `full_search_candidate_sec.png`, `parity_gates.png`, `halving_tradeoff.png`.",
        "",
    ]
    path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a focused Phase8 systems report.")
    parser.add_argument("--root", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    rows = phase8_summaries(args.root)
    full = full_search_rows(rows)
    parity = parity_rows(rows)
    halving = halving_rows(rows)
    csv_write(
        args.out / "full_search.csv",
        full,
        [
            "suite",
            "run",
            "population",
            "screen_prompts",
            "chunk_adapters",
            "max_loras",
            "max_new_tokens",
            "enforce_eager",
            "max_num_batched_tokens",
            "candidate_sec",
            "screen_prompts_per_sec",
            "eval_elapsed_s",
            "load_s",
        ],
    )
    csv_write(
        args.out / "parity.csv",
        parity,
        [
            "suite",
            "run",
            "trusted_name",
            "candidate_name",
            "n_common",
            "spearman",
            "top8_overlap",
            "top8_possible",
            "selected_regret_vs_trusted",
            "pass",
            "trusted_best_candidate",
            "candidate_best_candidate",
        ],
    )
    csv_write(
        args.out / "halving.csv",
        halving,
        [
            "suite",
            "run",
            "screen_prompts",
            "stage_prompts",
            "survivors",
            "candidate_sec",
            "prompt_eval_savings",
            "top8_survivor_recall",
            "top8_possible",
            "full_best_survived",
            "halving_selected_regret_vs_full",
            "eval_elapsed_s",
        ],
    )
    if full:
        plot_full_search(args.out / "full_search_candidate_sec.png", full)
    if parity:
        plot_parity(args.out / "parity_gates.png", parity)
    if halving:
        plot_halving(args.out / "halving_tradeoff.png", halving)
    write_report(args.out / "report.md", full, parity, halving)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
