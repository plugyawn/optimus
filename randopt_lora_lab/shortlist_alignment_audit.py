from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .compare_backends import pearson, spearman
from .parity_report import candidate_spec_key


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    return default if value is None else float(value)


def spec(row: dict[str, Any]) -> str:
    return candidate_spec_key(str(row["candidate"]))


def by_spec(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {spec(row): row for row in rows}


def sorted_specs(rows_by_spec: dict[str, dict[str, Any]], score_col: str) -> list[str]:
    return sorted(rows_by_spec, key=lambda key: (f(rows_by_spec[key], score_col), key), reverse=True)


def rank_map(order: list[str]) -> dict[str, int]:
    return {key: idx + 1 for idx, key in enumerate(order)}


def alignment(
    left: dict[str, dict[str, Any]],
    right: dict[str, dict[str, Any]],
    *,
    left_col: str,
    right_col: str,
) -> dict[str, Any]:
    common = sorted(set(left) & set(right))
    out: dict[str, Any] = {
        "common": len(common),
        "left_col": left_col,
        "right_col": right_col,
    }
    if len(common) < 2:
        out.update({"spearman": None, "pearson": None, "mean_abs_delta": None, "max_abs_delta": None})
        return out
    xs = [f(left[key], left_col) for key in common]
    ys = [f(right[key], right_col) for key in common]
    deltas = [abs(x - y) for x, y in zip(xs, ys)]
    out.update(
        {
            "spearman": spearman(xs, ys),
            "pearson": pearson(xs, ys),
            "mean_abs_delta": mean(deltas),
            "max_abs_delta": max(deltas),
            "left_mean": mean(xs),
            "right_mean": mean(ys),
        }
    )
    return out


def top_overlap(
    left: dict[str, dict[str, Any]],
    right: dict[str, dict[str, Any]],
    *,
    left_col: str,
    right_col: str,
    ks: list[int],
) -> list[dict[str, Any]]:
    common = sorted(set(left) & set(right))
    left_common = {key: left[key] for key in common}
    right_common = {key: right[key] for key in common}
    left_order = sorted_specs(left_common, left_col)
    right_order = sorted_specs(right_common, right_col)
    rows = []
    for k in ks:
        k_eff = min(k, len(common))
        rows.append(
            {
                "k": k,
                "possible": k_eff,
                "overlap": len(set(left_order[:k_eff]) & set(right_order[:k_eff])),
                "left_top": left_order[:k_eff],
                "right_top": right_order[:k_eff],
            }
        )
    return rows


def condition_map(rows: list[dict[str, Any]], variant: str) -> dict[str, dict[str, Any]]:
    return {spec(row): row for row in rows if str(row.get("prompt_variant")) == variant}


def dense_recall(
    dense: dict[str, dict[str, Any]],
    proposal: dict[str, dict[str, Any]],
    *,
    proposal_col: str,
    dense_col: str,
    ks: list[int],
) -> dict[str, Any]:
    common = sorted(set(dense) & set(proposal))
    dense_common = {key: dense[key] for key in common}
    proposal_common = {key: proposal[key] for key in common}
    dense_order = sorted_specs(dense_common, dense_col)
    proposal_order = sorted_specs(proposal_common, proposal_col)
    proposal_rank = rank_map(proposal_order)
    dense_best = dense_order[0] if dense_order else None
    out: dict[str, Any] = {
        "common": len(common),
        "dense_col": dense_col,
        "proposal_col": proposal_col,
        "dense_best_spec": dense_best,
        "dense_best_score": None if dense_best is None else f(dense[dense_best], dense_col),
        "dense_best_proposal_rank": None if dense_best is None else proposal_rank.get(dense_best),
        "dense_best_proposal_score": None if dense_best is None else f(proposal[dense_best], proposal_col),
    }
    out["rows"] = []
    dense_sets = {k: set(dense_order[: min(k, len(dense_order))]) for k in ks}
    for k in ks:
        k_eff = min(k, len(proposal_order))
        selected = set(proposal_order[:k_eff])
        out["rows"].append(
            {
                "k": k,
                "possible": k_eff,
                "contains_dense_best": dense_best in selected if dense_best else False,
                "dense_topk_overlap": len(selected & dense_sets[k]),
            }
        )
    return out


def per_prompt_alignment(vllm_rows: list[dict[str, Any]], peft_rows: list[dict[str, Any]]) -> dict[str, Any]:
    vllm = {
        (spec(row), int(row["example_id"])): row
        for row in vllm_rows
        if row.get("candidate") != "base" and row.get("mode") == "screen" and str(row.get("prompt_variant")) == "default"
    }
    peft = {
        (spec(row), int(row["example_id"])): row
        for row in peft_rows
        if row.get("candidate") != "base" and row.get("mode") == "screen"
    }
    keys = sorted(set(vllm) & set(peft))
    out: dict[str, Any] = {"common_rows": len(keys)}
    if not keys:
        return out
    exact_equal = [float(vllm[key].get("exact", 0.0)) == float(peft[key].get("exact", 0.0)) for key in keys]
    text_equal = [str(vllm[key].get("text", "")) == str(peft[key].get("text", "")) for key in keys]
    out.update(
        {
            "exact_equal_fraction": sum(exact_equal) / len(exact_equal),
            "text_equal_fraction": sum(text_equal) / len(text_equal),
            "vllm_exact_mean": mean([f(vllm[key], "exact") for key in keys]),
            "peft_exact_mean": mean([f(peft[key], "exact") for key in keys]),
        }
    )
    grouped: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for key in keys:
        grouped[key[0]].append(key)
    candidate_rows = []
    for key, candidate_keys in sorted(grouped.items()):
        candidate_rows.append(
            {
                "spec": key,
                "rows": len(candidate_keys),
                "exact_equal_fraction": sum(
                    float(vllm[item].get("exact", 0.0)) == float(peft[item].get("exact", 0.0)) for item in candidate_keys
                )
                / len(candidate_keys),
                "text_equal_fraction": sum(
                    str(vllm[item].get("text", "")) == str(peft[item].get("text", "")) for item in candidate_keys
                )
                / len(candidate_keys),
                "vllm_exact_mean": mean([f(vllm[item], "exact") for item in candidate_keys]),
                "peft_exact_mean": mean([f(peft[item], "exact") for item in candidate_keys]),
            }
        )
    out["candidate_rows"] = candidate_rows
    return out


def analyze(root: Path, *, ks: list[int]) -> dict[str, Any]:
    dense_rows = read_jsonl(root / "dense" / "candidate_summary.jsonl")
    proposal_rows = read_jsonl(root / "vllm" / "candidate_summary.jsonl")
    confirmed_rows = read_jsonl(root / "confirmed" / "candidate_summary.jsonl")
    condition_rows = read_jsonl(root / "vllm" / "candidate_condition_summary.jsonl")
    vllm_prompt_rows = read_jsonl(root / "vllm" / "per_prompt.jsonl")
    peft_prompt_rows = read_jsonl(root / "confirmed" / "per_prompt.jsonl")

    dense = by_spec(dense_rows)
    proposal = by_spec(proposal_rows)
    confirmed = by_spec(confirmed_rows)
    default = condition_map(condition_rows, "default")
    reordered = condition_map(condition_rows, "reordered")

    return {
        "kind": "shortlist_alignment_audit",
        "run_root": str(root),
        "counts": {
            "dense": len(dense_rows),
            "proposal": len(proposal_rows),
            "confirmed": len(confirmed_rows),
            "proposal_conditions": len(condition_rows),
        },
        "dense_vs_proposal": {
            "selection_score": alignment(dense, proposal, left_col="exact_mean", right_col="selection_score"),
            "proposal_exact": alignment(dense, proposal, left_col="exact_mean", right_col="exact_mean"),
            "default_exact": alignment(dense, default, left_col="exact_mean", right_col="exact_mean"),
            "reordered_exact": alignment(dense, reordered, left_col="exact_mean", right_col="exact_mean"),
            "dense_recall_by_selection": dense_recall(
                dense, proposal, proposal_col="selection_score", dense_col="exact_mean", ks=ks
            ),
            "dense_recall_by_default_exact": dense_recall(
                dense, default, proposal_col="exact_mean", dense_col="exact_mean", ks=ks
            ),
        },
        "confirmed_vs_proposal": {
            "selection_score": alignment(confirmed, proposal, left_col="exact_mean", right_col="selection_score"),
            "proposal_exact": alignment(confirmed, proposal, left_col="exact_mean", right_col="exact_mean"),
            "default_exact": alignment(confirmed, default, left_col="exact_mean", right_col="exact_mean"),
            "reordered_exact": alignment(confirmed, reordered, left_col="exact_mean", right_col="exact_mean"),
            "top_overlap_default": top_overlap(confirmed, default, left_col="exact_mean", right_col="exact_mean", ks=ks),
        },
        "vllm_default_vs_reordered": {
            "exact": alignment(default, reordered, left_col="exact_mean", right_col="exact_mean"),
            "selection": alignment(default, reordered, left_col="condition_selection_score", right_col="condition_selection_score"),
            "top_overlap": top_overlap(
                default, reordered, left_col="condition_selection_score", right_col="condition_selection_score", ks=ks
            ),
        },
        "per_prompt_default_vllm_vs_peft": per_prompt_alignment(vllm_prompt_rows, peft_prompt_rows),
    }


def fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Shortlist Alignment Audit",
        "",
        f"Run root: `{summary['run_root']}`",
        "",
        "## Counts",
        "",
        "| item | count |",
        "| --- | ---: |",
    ]
    for key, value in summary["counts"].items():
        lines.append(f"| {key} | {value} |")

    def metric_table(title: str, block: dict[str, Any]) -> None:
        lines.extend(["", f"## {title}", "", "| metric | common | spearman | pearson | mean abs delta | max abs delta |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
        for key, row in block.items():
            if not isinstance(row, dict) or "common" not in row or "spearman" not in row:
                continue
            lines.append(
                f"| {key} | {row['common']} | {fmt(row['spearman'])} | {fmt(row['pearson'])} | {fmt(row['mean_abs_delta'])} | {fmt(row['max_abs_delta'])} |"
            )

    metric_table("Dense vs vLLM Proposal", summary["dense_vs_proposal"])
    recall = summary["dense_vs_proposal"]["dense_recall_by_selection"]
    lines.extend(
        [
            "",
            "### Dense Recall By vLLM Selection",
            "",
            f"Dense best spec: `{recall['dense_best_spec']}`",
            f"Dense best proposal rank: `{recall['dense_best_proposal_rank']}`",
            "",
            "| k | contains dense best | dense top-k overlap |",
            "| ---: | --- | ---: |",
        ]
    )
    for row in recall["rows"]:
        lines.append(f"| {row['k']} | {str(row['contains_dense_best']).lower()} | {row['dense_topk_overlap']} |")

    metric_table("Confirmed PEFT vs vLLM Proposal", summary["confirmed_vs_proposal"])
    metric_table("vLLM Default vs Reordered", summary["vllm_default_vs_reordered"])

    pp = summary["per_prompt_default_vllm_vs_peft"]
    lines.extend(
        [
            "",
            "## Per-Prompt Default Backend Agreement",
            "",
            "| metric | value |",
            "| --- | ---: |",
            f"| common rows | {pp.get('common_rows', 0)} |",
            f"| exact equal fraction | {fmt(pp.get('exact_equal_fraction'))} |",
            f"| text equal fraction | {fmt(pp.get('text_equal_fraction'))} |",
            f"| vLLM exact mean | {fmt(pp.get('vllm_exact_mean'))} |",
            f"| PEFT exact mean | {fmt(pp.get('peft_exact_mean'))} |",
            "",
            "## Interpretation",
            "",
            "This audit separates the shortlist failure into three checks:",
            "",
            "1. Whether vLLM proposal scores agree with dense scores across the full population.",
            "2. Whether vLLM proposal scores agree with trusted PEFT scores on the confirmed shortlist.",
            "3. Whether default-prompt vLLM and PEFT outputs are identical enough to trust vLLM as a selector.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_ks(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit vLLM shortlist ranking alignment against dense and PEFT artifacts.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ks", default="1,2,4,8,16,32")
    args = parser.parse_args(argv)

    summary = analyze(args.root, ks=parse_ks(args.ks))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (args.out / "report.md").write_text(render_report(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
