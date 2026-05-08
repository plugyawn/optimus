from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .selector_calibration_audit import SELECTORS, load_panel, read_jsonl, with_builtin_scores


POLICIES: dict[str, list[str]] = {
    "current_selection": ["selection_score"],
    "proposal_exact": ["proposal_exact"],
    "default_exact": ["default_exact"],
    "prompt_exact_rr": ["default_exact", "reordered_exact", "xml_exact", "proposal_exact"],
    "prompt_lift_rr": ["default_lift", "reordered_lift", "xml_lift", "valid_mean_lift", "valid_min_lift"],
    "stability_rr": [
        "valid_mean_lift",
        "valid_min_lift",
        "low_spread_valid_mean",
        "default_minus_instability",
        "mean_minus_malformed",
    ],
    "all_builtin_rr": list(dict.fromkeys(SELECTORS.values())),
}


def is_finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def score_order(rows: list[dict[str, Any]], score_key: str) -> list[dict[str, Any]]:
    usable = [row for row in rows if is_finite(row.get(score_key))]
    return sorted(usable, key=lambda row: (float(row[score_key]), str(row["spec"])), reverse=True)


def dense_order(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (float(row["dense_exact"]), str(row["spec"])), reverse=True)


def round_robin_order(rows: list[dict[str, Any]], score_keys: list[str]) -> list[dict[str, Any]]:
    orders = [score_order(rows, key) for key in score_keys]
    max_len = max((len(order) for order in orders), default=0)
    by_spec = {row["spec"]: row for row in rows}
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for idx in range(max_len):
        for order in orders:
            if idx >= len(order):
                continue
            key = str(order[idx]["spec"])
            if key not in seen:
                seen.add(key)
                out.append(by_spec[key])
    return out


def evaluate_order(rows: list[dict[str, Any]], order: list[dict[str, Any]], *, ks: list[int]) -> dict[str, Any]:
    dense_ranked = dense_order(rows)
    dense_best = dense_ranked[0] if dense_ranked else None
    dense_best_spec = None if dense_best is None else str(dense_best["spec"])
    dense_best_score = None if dense_best is None else float(dense_best["dense_exact"])
    rank = {str(row["spec"]): idx + 1 for idx, row in enumerate(order)}
    dense_sets = {k: {str(row["spec"]) for row in dense_ranked[: min(k, len(dense_ranked))]} for k in ks}
    out: dict[str, Any] = {
        "candidate_count": len(rows),
        "dense_best_spec": dense_best_spec,
        "dense_best_score": dense_best_score,
        "dense_best_rank": None if dense_best_spec is None else rank.get(dense_best_spec),
        "rows": [],
    }
    for k in ks:
        selected = order[: min(k, len(order))]
        selected_specs = {str(row["spec"]) for row in selected}
        selected_best_dense = max((float(row["dense_exact"]) for row in selected), default=None)
        regret = None
        if dense_best_score is not None and selected_best_dense is not None:
            regret = dense_best_score - selected_best_dense
        out["rows"].append(
            {
                "k": k,
                "selected_count": len(selected),
                "contains_dense_best": dense_best_spec in selected_specs if dense_best_spec else False,
                "dense_topk_overlap": len(selected_specs & dense_sets[k]),
                "selected_best_dense": selected_best_dense,
                "dense_regret": regret,
                "selected_specs": [str(row["spec"]) for row in selected],
            }
        )
    return out


def analyze(runs: list[Path], *, ks: list[int]) -> dict[str, Any]:
    panels = []
    for run in runs:
        panel = load_panel(run)
        panels.append((panel.name, panel.root, with_builtin_scores(panel.rows)))

    per_run: dict[str, Any] = {}
    aggregate: dict[str, dict[int, list[dict[str, Any]]]] = {name: {k: [] for k in ks} for name in POLICIES}
    for name, root, rows in panels:
        policy_results = {}
        for policy_name, score_keys in POLICIES.items():
            order = round_robin_order(rows, score_keys)
            result = evaluate_order(rows, order, ks=ks)
            result["score_keys"] = score_keys
            policy_results[policy_name] = result
            for row in result["rows"]:
                aggregate[policy_name][int(row["k"])].append(row)
        per_run[name] = {
            "root": str(root),
            "candidate_count": len(rows),
            "policies": policy_results,
        }

    policy_summary: dict[str, Any] = {}
    for policy_name, by_k in aggregate.items():
        rows = []
        for k in ks:
            items = by_k[k]
            regrets = [float(row["dense_regret"]) for row in items if row["dense_regret"] is not None]
            rows.append(
                {
                    "k": k,
                    "run_count": len(items),
                    "dense_best_recall": sum(1 for row in items if row["contains_dense_best"]),
                    "dense_best_recall_rate": (sum(1 for row in items if row["contains_dense_best"]) / len(items)) if items else None,
                    "mean_dense_regret": (sum(regrets) / len(regrets)) if regrets else None,
                    "max_dense_regret": max(regrets) if regrets else None,
                    "mean_dense_topk_overlap": (sum(float(row["dense_topk_overlap"]) for row in items) / len(items)) if items else None,
                }
            )
        policy_summary[policy_name] = {"score_keys": POLICIES[policy_name], "rows": rows}

    best_by_k = {}
    for k in ks:
        best_by_k[str(k)] = sorted(
            (
                {
                    "policy": policy,
                    **next(row for row in payload["rows"] if int(row["k"]) == k),
                }
                for policy, payload in policy_summary.items()
            ),
            key=lambda row: (
                -float(row["dense_best_recall"]),
                float("inf") if row["max_dense_regret"] is None else float(row["max_dense_regret"]),
                float("inf") if row["mean_dense_regret"] is None else float(row["mean_dense_regret"]),
                row["policy"],
            ),
        )[:5]

    return {
        "kind": "selector_union_audit",
        "runs": [str(run) for run in runs],
        "ks": ks,
        "policies": POLICIES,
        "per_run": per_run,
        "policy_summary": policy_summary,
        "best_by_k": best_by_k,
        "verdict": verdict(policy_summary, ks=ks, run_count=len(panels)),
    }


def shortlist_for_run(root: Path, *, policy: str, k: int) -> list[dict[str, Any]]:
    if policy not in POLICIES:
        raise KeyError(f"unknown policy {policy!r}; choose one of {sorted(POLICIES)}")
    panel = load_panel(root)
    rows = with_builtin_scores(panel.rows)
    order = round_robin_order(rows, POLICIES[policy])
    candidate_rows = {}
    for row in read_jsonl(root / "vllm" / "candidate_summary.jsonl"):
        candidate_rows[str(row["candidate"])] = row
    out = []
    for rank, row in enumerate(order[: min(k, len(order))], start=1):
        candidate = str(row["candidate"])
        selected = dict(candidate_rows[candidate])
        selected["selector_union_policy"] = policy
        selected["selector_union_rank"] = rank
        selected["selector_union_score_keys"] = POLICIES[policy]
        selected["selector_union_dense_exact_offline"] = row["dense_exact"]
        out.append(selected)
    return out


def verdict(policy_summary: dict[str, Any], *, ks: list[int], run_count: int) -> dict[str, Any]:
    passing = []
    regret_thresholds = {
        "zero": 0.0,
        "one_screen_example": 1.0 / 64.0,
    }
    regret_passing: dict[str, Any] = {}
    for policy_name, payload in policy_summary.items():
        for row in payload["rows"]:
            if int(row["dense_best_recall"]) == run_count:
                passing.append(
                    {
                        "policy": policy_name,
                        "k": row["k"],
                        "max_dense_regret": row["max_dense_regret"],
                        "mean_dense_regret": row["mean_dense_regret"],
                    }
                )
            for label, threshold in regret_thresholds.items():
                max_regret = row["max_dense_regret"]
                if max_regret is not None and float(max_regret) <= threshold:
                    regret_passing.setdefault(label, []).append(
                        {
                            "policy": policy_name,
                            "k": row["k"],
                            "max_dense_regret": max_regret,
                            "mean_dense_regret": row["mean_dense_regret"],
                            "dense_best_recall": row["dense_best_recall"],
                        }
                    )
    passing = sorted(
        passing,
        key=lambda row: (
            int(row["k"]),
            float("inf") if row["max_dense_regret"] is None else float(row["max_dense_regret"]),
            row["policy"],
        ),
    )
    regret_first = {}
    for label, rows in regret_passing.items():
        rows = sorted(
            rows,
            key=lambda row: (
                int(row["k"]),
                -int(row["dense_best_recall"]),
                float("inf") if row["mean_dense_regret"] is None else float(row["mean_dense_regret"]),
                row["policy"],
            ),
        )
        regret_first[label] = rows[0] if rows else None
    first = passing[0] if passing else None
    return {
        "run_count": run_count,
        "first_policy_recovering_dense_best_all_runs": first,
        "first_policy_with_max_regret_at_most": regret_first,
        "pass_at_k8": bool(first and int(first["k"]) <= 8),
        "pass_at_k16": bool(first and int(first["k"]) <= 16),
        "interpretation": (
            "This is an offline dense-recall prerequisite. A pass means a vLLM-score policy would "
            "have sent the dense PEFT screen winner to confirmation on every saved panel."
        ),
    }


def fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Selector Union Audit",
        "",
        f"Runs: `{', '.join(summary['runs'])}`",
        "",
        "## Verdict",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| run count | {summary['verdict']['run_count']} |",
        f"| first all-run dense-best policy | {summary['verdict']['first_policy_recovering_dense_best_all_runs']} |",
        f"| first all-run zero-regret policy | {summary['verdict']['first_policy_with_max_regret_at_most'].get('zero')} |",
        f"| first all-run <=1/64 screen-regret policy | {summary['verdict']['first_policy_with_max_regret_at_most'].get('one_screen_example')} |",
        f"| pass at k<=8 | {str(summary['verdict']['pass_at_k8']).lower()} |",
        f"| pass at k<=16 | {str(summary['verdict']['pass_at_k16']).lower()} |",
        "",
        "## Aggregate Policy Recall",
        "",
        "| policy | k | dense-best recall | mean regret | max regret | mean top-k overlap |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for policy_name, payload in sorted(summary["policy_summary"].items()):
        for row in payload["rows"]:
            lines.append(
                f"| {policy_name} | {row['k']} | {row['dense_best_recall']}/{row['run_count']} | "
                f"{fmt(row['mean_dense_regret'])} | {fmt(row['max_dense_regret'])} | "
                f"{fmt(row['mean_dense_topk_overlap'])} |"
            )

    lines.extend(["", "## Per-Run Dense-Best Rank", "", "| run | policy | dense best rank | k=8 contains best | k=16 contains best |", "| --- | --- | ---: | --- | --- |"])
    for run_name, payload in summary["per_run"].items():
        for policy_name, result in sorted(payload["policies"].items()):
            row8 = next((row for row in result["rows"] if int(row["k"]) == 8), None)
            row16 = next((row for row in result["rows"] if int(row["k"]) == 16), None)
            lines.append(
                f"| {run_name} | {policy_name} | {fmt(result['dense_best_rank'])} | "
                f"{None if row8 is None else str(row8['contains_dense_best']).lower()} | "
                f"{None if row16 is None else str(row16['contains_dense_best']).lower()} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This audit does not replace PEFT confirmation. It tests whether a cheap vLLM "
            "shortlist policy has enough dense-winner recall to justify spending PEFT "
            "confirmation budget on its selected candidates.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(summary: dict[str, Any], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (out / "report.md").write_text(render_report(summary))
    with (out / "policy_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "policy",
                "k",
                "run_count",
                "dense_best_recall",
                "dense_best_recall_rate",
                "mean_dense_regret",
                "max_dense_regret",
                "mean_dense_topk_overlap",
            ],
        )
        writer.writeheader()
        for policy, payload in summary["policy_summary"].items():
            for row in payload["rows"]:
                writer.writerow({"policy": policy, **row})


def parse_ks(text: str) -> list[int]:
    return [int(item) for item in text.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit or write round-robin union policies for vLLM selector dense recall.")
    sub = parser.add_subparsers(dest="cmd")

    audit = sub.add_parser("audit", help="run a multi-panel offline selector audit")
    audit.add_argument("--run", type=Path, action="append", required=True)
    audit.add_argument("--out", type=Path, required=True)
    audit.add_argument("--ks", default="4,8,16,32")

    shortlist = sub.add_parser("shortlist", help="write a shortlist JSONL for one run and policy")
    shortlist.add_argument("--run", type=Path, required=True)
    shortlist.add_argument("--out", type=Path, required=True)
    shortlist.add_argument("--policy", choices=sorted(POLICIES), required=True)
    shortlist.add_argument("--k", type=int, required=True)

    args = parser.parse_args()
    if args.cmd in (None, "audit"):
        summary = analyze(args.run, ks=parse_ks(args.ks))
        write_outputs(summary, args.out)
        print(json.dumps(summary["verdict"], indent=2, sort_keys=True))
    elif args.cmd == "shortlist":
        rows = shortlist_for_run(args.run, policy=args.policy, k=args.k)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        print(json.dumps({"out": str(args.out), "policy": args.policy, "written": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
