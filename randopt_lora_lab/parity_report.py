from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def candidate_spec_key(candidate_key: str) -> str:
    parts = candidate_key.split(":")
    if len(parts) != 4:
        raise ValueError(f"invalid candidate key: {candidate_key}")
    return ":".join(parts[1:])


def average_ranks(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=np.float64)
    i = 0
    while i < len(arr):
        j = i + 1
        while j < len(arr) and arr[order[j]] == arr[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    rx = average_ranks(xs)
    ry = average_ranks(ys)
    if float(rx.std()) == 0.0 or float(ry.std()) == 0.0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def load_run(run_dir: Path) -> dict:
    summary_path = run_dir / "summary.json"
    candidates_path = run_dir / "candidate_summary.jsonl"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    if not candidates_path.exists():
        raise FileNotFoundError(candidates_path)
    return {
        "summary": json.loads(summary_path.read_text()),
        "candidates": read_jsonl(candidates_path),
    }


def compare_runs(
    dense_run: dict,
    lora_run: dict,
    *,
    top_k: int = 8,
    min_spearman: float = 0.85,
    min_topk_overlap: int = 6,
    max_selected_regret: float = 0.0,
) -> dict:
    dense_by_key = {candidate_spec_key(row["candidate"]): row for row in dense_run["candidates"]}
    lora_by_key = {candidate_spec_key(row["candidate"]): row for row in lora_run["candidates"]}
    shared_keys = sorted(set(dense_by_key) & set(lora_by_key))
    dense_scores = [float(dense_by_key[key]["exact_mean"]) for key in shared_keys]
    lora_scores = [float(lora_by_key[key]["exact_mean"]) for key in shared_keys]
    dense_ranked = sorted(shared_keys, key=lambda key: dense_by_key[key]["exact_mean"], reverse=True)
    lora_ranked = sorted(shared_keys, key=lambda key: lora_by_key[key]["exact_mean"], reverse=True)
    dense_top = set(dense_ranked[:top_k])
    lora_top = set(lora_ranked[:top_k])
    dense_best_key = dense_ranked[0] if dense_ranked else None
    lora_pick_key = lora_ranked[0] if lora_ranked else None
    dense_best_score = float(dense_by_key[dense_best_key]["exact_mean"]) if dense_best_key else None
    dense_score_at_lora_pick = float(dense_by_key[lora_pick_key]["exact_mean"]) if lora_pick_key else None
    selected_regret = None
    if dense_best_score is not None and dense_score_at_lora_pick is not None:
        selected_regret = dense_best_score - dense_score_at_lora_pick
    dense_candidate_sec = dense_run["summary"].get("candidate_sec")
    lora_candidate_sec = lora_run["summary"].get("candidate_sec")
    speed_ratio = None
    if dense_candidate_sec and lora_candidate_sec:
        speed_ratio = float(lora_candidate_sec) / float(dense_candidate_sec)
    rho = spearman(dense_scores, lora_scores)
    topk_overlap = len(dense_top & lora_top)
    gates = {
        "shared_panel": len(shared_keys) == len(dense_run["candidates"]) == len(lora_run["candidates"]),
        "spearman": rho is not None and rho >= min_spearman,
        "topk_overlap": topk_overlap >= min_topk_overlap,
        "selected_regret": selected_regret is not None and selected_regret <= max_selected_regret,
        "speed": speed_ratio is not None and speed_ratio >= 1.0,
    }
    return {
        "shared_candidates": len(shared_keys),
        "dense_candidates": len(dense_run["candidates"]),
        "lora_candidates": len(lora_run["candidates"]),
        "spearman": rho,
        "top_k": top_k,
        "topk_overlap": topk_overlap,
        "dense_best_key": dense_best_key,
        "lora_pick_key": lora_pick_key,
        "dense_best_score": dense_best_score,
        "dense_score_at_lora_pick": dense_score_at_lora_pick,
        "selected_regret": selected_regret,
        "dense_candidate_sec": dense_candidate_sec,
        "lora_candidate_sec": lora_candidate_sec,
        "speed_ratio_lora_over_dense": speed_ratio,
        "thresholds": {
            "min_spearman": min_spearman,
            "min_topk_overlap": min_topk_overlap,
            "max_selected_regret": max_selected_regret,
        },
        "gates": gates,
        "pass": all(gates.values()),
    }


def render_markdown(summary: dict) -> str:
    lines = [
        "# Dense Gaussian vs LoRA Parity Report",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| shared candidates | {summary['shared_candidates']} |",
        f"| Spearman | {summary['spearman'] if summary['spearman'] is not None else 'null'} |",
        f"| top-{summary['top_k']} overlap | {summary['topk_overlap']} |",
        f"| selected regret | {summary['selected_regret'] if summary['selected_regret'] is not None else 'null'} |",
        f"| dense candidate/sec | {summary['dense_candidate_sec']} |",
        f"| LoRA candidate/sec | {summary['lora_candidate_sec']} |",
        f"| speed ratio LoRA/dense | {summary['speed_ratio_lora_over_dense'] if summary['speed_ratio_lora_over_dense'] is not None else 'null'} |",
        "",
        "## Gates",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for gate, passed in summary["gates"].items():
        lines.append(f"| {gate} | {str(passed).lower()} |")
    lines.extend([
        "",
        f"Overall pass: `{str(summary['pass']).lower()}`",
        "",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare dense Gaussian and LoRA search result parity.")
    parser.add_argument("--dense", type=Path, required=True, help="Dense Gaussian run directory")
    parser.add_argument("--lora", type=Path, required=True, help="LoRA run directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--min-spearman", type=float, default=0.85)
    parser.add_argument("--min-topk-overlap", type=int, default=6)
    parser.add_argument("--max-selected-regret", type=float, default=0.0)
    args = parser.parse_args(argv)

    summary = compare_runs(
        load_run(args.dense),
        load_run(args.lora),
        top_k=args.top_k,
        min_spearman=args.min_spearman,
        min_topk_overlap=args.min_topk_overlap,
        max_selected_regret=args.max_selected_regret,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (args.out / "report.md").write_text(render_markdown(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
