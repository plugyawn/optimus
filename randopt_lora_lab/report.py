from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


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


def max_present(rows: list[dict], key: str):
    values = [x[key] for x in rows if key in x and x[key] is not None]
    return max(values) if values else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for path in root.glob("*/summary.json"):
        row = json.loads(path.read_text())
        row["run"] = path.parent.name
        summaries.append(row)
    flat = []
    for row in summaries:
        top_holdout = row.get("top_holdout") or []
        best_holdout = max((x.get("exact_mean", 0.0) for x in top_holdout), default=None)
        tokens_per_sec = (
            row.get("mixed_tokens_per_sec")
            or row.get("lora_tokens_per_sec")
            or row.get("best_tokens_per_sec")
            or row.get("tokens_per_sec")
        )
        prompts_per_sec = (
            row.get("mixed_prompts_per_sec")
            or row.get("lora_prompts_per_sec")
            or row.get("best_prompts_per_sec")
            or row.get("prompts_per_sec")
        )
        if row.get("mixed_tokens_per_sec"):
            throughput_mode = "mixed_lora"
        elif row.get("lora_tokens_per_sec"):
            throughput_mode = "sequential_lora"
        else:
            throughput_mode = "native"
        flat.append(
            {
                "run": row["run"],
                "kind": row.get("kind"),
                "family": row.get("family", ""),
                "population": row.get("population", 0),
                "stop_at_answer": row.get("stop_at_answer"),
                "max_new_tokens": row.get("max_new_tokens"),
                "base_screen_exact": row.get("base_screen_exact", row.get("base_exact")),
                "best_holdout_exact": best_holdout,
                "best_cap_hit_mean": max_present(top_holdout, "cap_hit_mean"),
                "best_answer_closed_mean": max_present(top_holdout, "answer_closed_mean"),
                "candidate_sec": row.get("candidate_sec"),
                "pair_sec": row.get("pair_sec"),
                "prompt_eval_savings": row.get("prompt_eval_savings"),
                "best_tokens_per_sec": tokens_per_sec,
                "best_prompts_per_sec": prompts_per_sec,
                "throughput_mode": throughput_mode,
                "best_batch_size": row.get("best_batch_size"),
            }
        )
    df = pd.DataFrame(flat)
    df.to_csv(out / "summary.csv", index=False)
    if "candidate_sec" in df and df["candidate_sec"].notna().any():
        ax = df.dropna(subset=["candidate_sec"]).plot.bar(x="run", y="candidate_sec", legend=False)
        ax.set_ylabel("candidates/sec")
        plt.tight_layout()
        plt.savefig(out / "candidate_sec.png", dpi=160)
        plt.close()
    if "best_tokens_per_sec" in df and df["best_tokens_per_sec"].notna().any():
        ax = df.dropna(subset=["best_tokens_per_sec"]).plot.bar(x="run", y="best_tokens_per_sec", legend=False)
        ax.set_ylabel("tokens/sec")
        plt.tight_layout()
        plt.savefig(out / "tokens_per_sec.png", dpi=160)
        plt.close()
    candidate_frames = []
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        rows = read_jsonl(run_dir / "candidate_summary.jsonl")
        if not rows:
            rows = read_jsonl(run_dir / "stage_candidate_summary.jsonl")
        if not rows:
            continue
        cdf = pd.DataFrame(rows)
        if "exact_mean" not in cdf:
            continue
        cdf["run"] = run_dir.name
        cdf["ordinal"] = range(1, len(cdf) + 1)
        cdf["best_so_far"] = cdf["exact_mean"].cummax()
        candidate_frames.append(cdf[["run", "candidate", "ordinal", "exact_mean", "best_so_far"]])
    if candidate_frames:
        candidates = pd.concat(candidate_frames, ignore_index=True)
        candidates.to_csv(out / "candidate_scores.csv", index=False)
        plt.figure(figsize=(9, 5))
        for run, g in candidates.groupby("run"):
            plt.plot(g["ordinal"], g["best_so_far"], label=run)
        plt.xlabel("candidates evaluated")
        plt.ylabel("best screen exact")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(out / "best_of_n.png", dpi=160)
        plt.close()
        pivot_runs = candidates["run"].unique()
        cols = min(3, len(pivot_runs))
        rows_n = (len(pivot_runs) + cols - 1) // cols
        fig, axes = plt.subplots(rows_n, cols, figsize=(4 * cols, 3 * rows_n), squeeze=False)
        for ax, run in zip(axes.ravel(), pivot_runs):
            g = candidates[candidates["run"] == run]
            ax.hist(g["exact_mean"], bins=8)
            ax.set_title(run, fontsize=8)
            ax.set_xlabel("screen exact")
        for ax in axes.ravel()[len(pivot_runs) :]:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(out / "score_histograms.png", dpi=160)
        plt.close()
    notes = []
    if "candidate_sec" in df and df["candidate_sec"].notna().any():
        best_candidate = df.dropna(subset=["candidate_sec"]).sort_values("candidate_sec", ascending=False).iloc[0]
        notes.append(
            f"- Fastest candidate-block run: `{best_candidate['run']}` at "
            f"{best_candidate['candidate_sec']:.3f} candidates/sec."
        )
    if "best_tokens_per_sec" in df and df["best_tokens_per_sec"].notna().any():
        best_tokens = df.dropna(subset=["best_tokens_per_sec"]).sort_values("best_tokens_per_sec", ascending=False).iloc[0]
        notes.append(
            f"- Fastest generation backend row: `{best_tokens['run']}` at "
            f"{best_tokens['best_tokens_per_sec']:.1f} tokens/sec."
        )
    body = "# RandOpt LoRA Lab Report\n\n"
    if notes:
        body += "## Auto Notes\n\n" + "\n".join(notes) + "\n\n"
    body += df.to_markdown(index=False) + "\n"
    (out / "report.md").write_text(body)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
