from __future__ import annotations

import argparse
import json
import math
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


def scalar(row: dict, key: str, default=None):
    value = row.get(key, default)
    return default if value is None else value


def wilson_interval(rate: float | None, n: int | None, z: float = 1.96) -> tuple[float | None, float | None]:
    if rate is None or not n:
        return None, None
    successes = max(0.0, min(float(n), float(rate) * n))
    denom = 1.0 + z * z / n
    center = (successes / n + z * z / (2 * n)) / denom
    margin = z * ((successes / n * (1.0 - successes / n) + z * z / (4 * n)) / n) ** 0.5 / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def binomial_tail_ge(k: int, n: int, p: float | None) -> float | None:
    if p is None or n <= 0:
        return None
    p = max(0.0, min(1.0, float(p)))
    k = max(0, min(n, int(k)))
    return sum(math.comb(n, i) * (p**i) * ((1.0 - p) ** (n - i)) for i in range(k, n + 1))


def best_of_n_null_p(rate: float | None, prompts: int | None, population: int | None, base_rate: float | None) -> float | None:
    if rate is None or not prompts or not population:
        return None
    successes = int(math.ceil(float(rate) * prompts - 1e-12))
    single = binomial_tail_ge(successes, prompts, base_rate)
    if single is None:
        return None
    return 1.0 - (1.0 - single) ** int(population)


def exact_binomial_sign_p(gains: int, losses: int) -> float | None:
    n = gains + losses
    if n == 0:
        return None
    tail = sum(math.comb(n, i) for i in range(0, min(gains, losses) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def paired_quality_metrics(run_dir: Path, top_holdout: list[dict]) -> dict:
    best = max(top_holdout, key=lambda x: x.get("exact_mean", 0.0), default=None)
    if not best:
        return {}
    best_candidate = best.get("candidate")
    rows = read_jsonl(run_dir / "holdout_per_prompt.jsonl")
    base_rows = [
        row
        for row in rows
        if row.get("candidate") == "base" and row.get("mode") in {None, "base_holdout"}
    ]
    candidate_rows = [
        row
        for row in rows
        if row.get("candidate") == best_candidate and row.get("mode") in {None, "holdout"}
    ]
    if not base_rows or not candidate_rows:
        return {"best_holdout_candidate": best_candidate, "paired_available": False}
    base_by_id = {row["example_id"]: row for row in base_rows}
    cand_by_id = {row["example_id"]: row for row in candidate_rows}
    common_ids = sorted(set(base_by_id) & set(cand_by_id))
    gains = losses = both = neither = 0
    deltas = []
    malformed_deltas = []
    cap_deltas = []
    for example_id in common_ids:
        base = base_by_id[example_id]
        cand = cand_by_id[example_id]
        b = float(base.get("exact", 0.0))
        c = float(cand.get("exact", 0.0))
        deltas.append(c - b)
        malformed_deltas.append(float(cand.get("malformed", 0.0)) - float(base.get("malformed", 0.0)))
        cap_deltas.append(float(cand.get("cap_hit", 0.0)) - float(base.get("cap_hit", 0.0)))
        if c > b:
            gains += 1
        elif b > c:
            losses += 1
        elif c > 0:
            both += 1
        else:
            neither += 1
    return {
        "best_holdout_candidate": best_candidate,
        "paired_available": bool(common_ids),
        "paired_n": len(common_ids),
        "paired_gain_count": gains,
        "paired_loss_count": losses,
        "paired_both_correct_count": both,
        "paired_neither_correct_count": neither,
        "paired_lift": None if not deltas else float(sum(deltas) / len(deltas)),
        "paired_sign_test_p": exact_binomial_sign_p(gains, losses),
        "paired_malformed_delta": None if not malformed_deltas else float(sum(malformed_deltas) / len(malformed_deltas)),
        "paired_cap_hit_delta": None if not cap_deltas else float(sum(cap_deltas) / len(cap_deltas)),
    }


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
        top_screen = row.get("top_screen") or []
        best_holdout = max((x.get("exact_mean", 0.0) for x in top_holdout), default=None)
        best_screen = max((x.get("exact_mean", 0.0) for x in top_screen), default=None)
        holdout_n = row.get("holdout_unique_prompts") or row.get("holdout_prompts")
        best_ci_low, best_ci_high = wilson_interval(best_holdout, holdout_n)
        base_holdout = row.get("base_holdout_exact")
        base_ci_low, base_ci_high = wilson_interval(base_holdout, holdout_n)
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
        paired = {
            "best_holdout_candidate": None,
            "paired_available": False,
            "paired_n": None,
            "paired_gain_count": None,
            "paired_loss_count": None,
            "paired_both_correct_count": None,
            "paired_neither_correct_count": None,
            "paired_lift": None,
            "paired_sign_test_p": None,
            "paired_malformed_delta": None,
            "paired_cap_hit_delta": None,
            **paired_quality_metrics(path.parent, top_holdout),
        }
        flat_row = {
                "run": row["run"],
                "kind": row.get("kind"),
                "family": row.get("family", ""),
                "population": row.get("population", 0),
                "screen_prompts": row.get("screen_prompts"),
                "holdout_prompts": row.get("holdout_prompts"),
                "stop_at_answer": row.get("stop_at_answer"),
                "max_new_tokens": row.get("max_new_tokens"),
                "base_screen_exact": row.get("base_screen_exact", row.get("base_exact")),
                "base_holdout_exact": row.get("base_holdout_exact"),
                "best_holdout_exact": best_holdout,
                "base_holdout_ci_low": base_ci_low,
                "base_holdout_ci_high": base_ci_high,
                "best_holdout_ci_low": best_ci_low,
                "best_holdout_ci_high": best_ci_high,
                "holdout_lift": None
                if best_holdout is None or row.get("base_holdout_exact") is None
                else best_holdout - row.get("base_holdout_exact"),
                "screen_unique_prompts": row.get("screen_unique_prompts"),
                "holdout_unique_prompts": row.get("holdout_unique_prompts"),
                "screen_unique_semantic_prompts": row.get("screen_unique_semantic_prompts"),
                "holdout_unique_semantic_prompts": row.get("holdout_unique_semantic_prompts"),
                "screen_holdout_overlap": row.get("screen_holdout_overlap"),
                "best_screen_exact": best_screen,
                "screen_best_of_n_null_p": best_of_n_null_p(
                    best_screen,
                    row.get("screen_unique_prompts") or row.get("screen_prompts"),
                    row.get("population") or row.get("population_total"),
                    row.get("base_screen_exact", row.get("base_exact")),
                ),
                "best_cap_hit_mean": max_present(top_holdout, "cap_hit_mean"),
                "best_answer_closed_mean": max_present(top_holdout, "answer_closed_mean"),
                "best_malformed_mean": max_present(top_holdout, "malformed_mean"),
                "candidate_sec": row.get("candidate_sec"),
                "pair_sec": row.get("pair_sec"),
                "prompt_eval_savings": row.get("prompt_eval_savings"),
                "best_tokens_per_sec": tokens_per_sec,
                "best_prompts_per_sec": prompts_per_sec,
                "throughput_mode": throughput_mode,
                "best_batch_size": row.get("best_batch_size"),
                "a_frob_mean": row.get("a_frob_mean"),
                "b_frob_mean": row.get("b_frob_mean"),
                "ba_frob_mean": row.get("ba_frob_mean"),
                "ba_frob_upper_mean": row.get("ba_frob_upper_mean"),
            }
        flat_row.update(paired)
        flat.append(flat_row)
    df = pd.DataFrame(flat)
    df.to_csv(out / "summary.csv", index=False)
    quality = df[df["best_holdout_exact"].notna()].copy()
    systems = df[df["best_tokens_per_sec"].notna()].copy()
    if not quality.empty:
        quality.sort_values("best_holdout_exact", ascending=False).to_csv(out / "quality_summary.csv", index=False)
        invalid = quality[
            (quality["screen_holdout_overlap"].fillna(0) > 0)
            | (quality["holdout_unique_prompts"].fillna(quality["holdout_prompts"]) < quality["holdout_prompts"])
            | (
                quality["holdout_unique_semantic_prompts"]
                .fillna(quality["holdout_unique_prompts"])
                .fillna(quality["holdout_prompts"])
                < quality["holdout_prompts"]
            )
        ]
        if not invalid.empty:
            invalid.to_csv(out / "quality_invalid_eval_rows.csv", index=False)
    if not systems.empty:
        systems.sort_values("best_tokens_per_sec", ascending=False).to_csv(out / "systems_summary.csv", index=False)
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
    if not quality.empty:
        plot_df = quality.sort_values("best_holdout_exact", ascending=False)
        ax = plot_df.plot.bar(x="run", y="best_holdout_exact", legend=False, figsize=(10, 4))
        ax.set_ylabel("best holdout exact")
        plt.tight_layout()
        plt.savefig(out / "quality_holdout.png", dpi=160)
        plt.close()
        if plot_df["holdout_lift"].notna().any():
            ax = plot_df.dropna(subset=["holdout_lift"]).plot.bar(x="run", y="holdout_lift", legend=False, figsize=(10, 4))
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.set_ylabel("holdout lift vs base")
            plt.tight_layout()
            plt.savefig(out / "quality_lift.png", dpi=160)
            plt.close()
        paired = quality[quality.get("paired_available", False) == True] if "paired_available" in quality else pd.DataFrame()
        if not paired.empty:
            paired.sort_values("paired_lift", ascending=False).to_csv(out / "paired_quality_summary.csv", index=False)
    if "best_cap_hit_mean" in df and df["best_cap_hit_mean"].notna().any():
        audit = df.dropna(subset=["best_cap_hit_mean"])
        ax = audit.plot.bar(x="run", y=["best_cap_hit_mean", "best_answer_closed_mean"], figsize=(10, 4))
        ax.set_ylabel("fraction")
        plt.tight_layout()
        plt.savefig(out / "cap_answer_audit.png", dpi=160)
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
    if not quality.empty:
        best_quality = quality.sort_values("best_holdout_exact", ascending=False).iloc[0]
        lift = best_quality["holdout_lift"]
        lift_text = "" if pd.isna(lift) else f" ({lift:+.4f} lift vs base)"
        notes.append(
            f"- Best holdout run: `{best_quality['run']}` at "
            f"{best_quality['best_holdout_exact']:.4f}{lift_text}."
        )
    if "prompt_eval_savings" in df and df["prompt_eval_savings"].notna().any():
        best_savings = df.dropna(subset=["prompt_eval_savings"]).sort_values("prompt_eval_savings", ascending=False).iloc[0]
        notes.append(
            f"- Best staged-screen saving: `{best_savings['run']}` saved "
            f"{100.0 * best_savings['prompt_eval_savings']:.1f}% of prompt evals."
        )
    mixed_rows = df[df["throughput_mode"] == "mixed_lora"]
    seq_rows = df[df["throughput_mode"] == "sequential_lora"]
    if not mixed_rows.empty and not seq_rows.empty:
        mixed = mixed_rows.sort_values("best_tokens_per_sec", ascending=False).iloc[0]
        seq = seq_rows.sort_values("best_tokens_per_sec", ascending=False).iloc[0]
        notes.append(
            f"- Mixed LoRA speedup vs best sequential LoRA row: "
            f"{mixed['best_tokens_per_sec'] / max(seq['best_tokens_per_sec'], 1e-9):.2f}x."
        )
    body = "# RandOpt LoRA Lab Report\n\n"
    if notes:
        body += "## Auto Notes\n\n" + "\n".join(notes) + "\n\n"
    if not systems.empty:
        body += "## Systems Top Rows\n\n"
        body += systems.sort_values("best_tokens_per_sec", ascending=False)[
            ["run", "kind", "throughput_mode", "best_tokens_per_sec", "best_prompts_per_sec"]
        ].head(8).to_markdown(index=False) + "\n\n"
    if not quality.empty:
        body += "## Research Top Rows\n\n"
        body += quality.sort_values("best_holdout_exact", ascending=False)[
            [
                "run",
                "kind",
                "family",
                "population",
                "base_holdout_exact",
                "best_holdout_exact",
                "holdout_lift",
                "paired_lift",
                "paired_gain_count",
                "paired_loss_count",
                "paired_sign_test_p",
                "screen_best_of_n_null_p",
                "best_holdout_ci_low",
                "best_holdout_ci_high",
                "screen_holdout_overlap",
                "prompt_eval_savings",
            ]
        ].head(12).to_markdown(index=False) + "\n\n"
    body += "## Full Summary\n\n"
    body += df.to_markdown(index=False) + "\n"
    (out / "report.md").write_text(body)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
