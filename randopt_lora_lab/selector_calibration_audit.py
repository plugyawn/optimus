from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from .compare_backends import pearson, spearman
from .parity_report import candidate_spec_key


VARIANTS = ("default", "reordered", "xml")


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


def score_sort_key(row: dict[str, Any], score_key: str) -> tuple[float, str]:
    return (float(row[score_key]), str(row["spec"]))


def order_by(rows: list[dict[str, Any]], score_key: str) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: score_sort_key(row, score_key), reverse=True)


def dense_order(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return order_by(rows, "dense_exact")


@dataclass(frozen=True)
class RunPanel:
    name: str
    root: Path
    rows: list[dict[str, Any]]


def load_panel(root: Path) -> RunPanel:
    dense_rows = {spec(row): row for row in read_jsonl(root / "dense" / "candidate_summary.jsonl")}
    proposal_rows = {spec(row): row for row in read_jsonl(root / "vllm" / "candidate_summary.jsonl")}
    condition_rows: dict[str, dict[str, dict[str, Any]]] = {}
    for row in read_jsonl(root / "vllm" / "candidate_condition_summary.jsonl"):
        condition_rows.setdefault(spec(row), {})[str(row.get("prompt_variant", "default"))] = row

    common = sorted(set(dense_rows) & set(proposal_rows))
    rows = []
    for key in common:
        dense = dense_rows[key]
        proposal = proposal_rows[key]
        by_variant = condition_rows.get(key, {})
        out: dict[str, Any] = {
            "run": root.name,
            "root": str(root),
            "spec": key,
            "candidate": proposal["candidate"],
            "dense_exact": f(dense, "exact_mean"),
            "proposal_exact": f(proposal, "exact_mean"),
            "selection_score": f(proposal, "selection_score"),
            "mean_condition_selection_score": f(proposal, "mean_condition_selection_score"),
            "min_condition_selection_score": f(proposal, "min_condition_selection_score"),
            "mean_exact_lift_vs_base": f(proposal, "mean_exact_lift_vs_base"),
            "min_exact_lift_vs_base": f(proposal, "min_exact_lift_vs_base"),
            "max_malformed_regression_vs_base": f(proposal, "max_malformed_regression_vs_base"),
            "max_cap_hit_regression_vs_base": f(proposal, "max_cap_hit_regression_vs_base"),
            "sigma": f(proposal, "sigma"),
            "sign": f(proposal, "sign"),
        }
        exacts = []
        lifts = []
        penalties = []
        for variant in VARIANTS:
            cond = by_variant.get(variant, {})
            exact = f(cond, "exact_mean", np.nan)
            lift = f(cond, "exact_lift_vs_base", np.nan)
            malformed = f(cond, "malformed_mean", np.nan)
            cap_hit = f(cond, "cap_hit_mean", np.nan)
            cond_score = f(cond, "condition_selection_score", np.nan)
            out[f"{variant}_exact"] = exact
            out[f"{variant}_lift"] = lift
            out[f"{variant}_condition_score"] = cond_score
            out[f"{variant}_malformed"] = malformed
            out[f"{variant}_cap_hit"] = cap_hit
            out[f"{variant}_output_tokens"] = f(cond, "output_tokens", np.nan)
            if not np.isnan(exact):
                exacts.append(exact)
            if not np.isnan(lift):
                lifts.append(lift)
            if not np.isnan(malformed) and not np.isnan(cap_hit):
                penalties.append(max(malformed, 0.0) + max(cap_hit, 0.0))
        valid_lifts = [out["default_lift"], out["reordered_lift"]]
        valid_lifts = [value for value in valid_lifts if not np.isnan(value)]
        valid_exacts = [out["default_exact"], out["reordered_exact"]]
        valid_exacts = [value for value in valid_exacts if not np.isnan(value)]
        out["valid_mean_exact"] = float(mean(valid_exacts)) if valid_exacts else np.nan
        out["valid_mean_lift"] = float(mean(valid_lifts)) if valid_lifts else np.nan
        out["valid_min_lift"] = float(min(valid_lifts)) if valid_lifts else np.nan
        out["valid_max_lift"] = float(max(valid_lifts)) if valid_lifts else np.nan
        out["valid_lift_spread"] = float(max(valid_lifts) - min(valid_lifts)) if valid_lifts else np.nan
        out["all_mean_exact"] = float(mean(exacts)) if exacts else np.nan
        out["all_mean_lift"] = float(mean(lifts)) if lifts else np.nan
        out["all_min_lift"] = float(min(lifts)) if lifts else np.nan
        out["all_max_lift"] = float(max(lifts)) if lifts else np.nan
        out["malformed_cap_penalty"] = float(max(penalties) if penalties else 0.0)
        rows.append(out)
    return RunPanel(name=root.name, root=root, rows=rows)


SELECTORS = {
    "current_selection": "selection_score",
    "proposal_exact": "proposal_exact",
    "default_exact": "default_exact",
    "reordered_exact": "reordered_exact",
    "xml_exact": "xml_exact",
    "valid_mean_exact": "valid_mean_exact",
    "valid_mean_lift": "valid_mean_lift",
    "valid_min_lift": "valid_min_lift",
    "valid_max_lift": "valid_max_lift",
    "low_spread_valid_mean": "low_spread_valid_mean",
    "default_minus_instability": "default_minus_instability",
    "mean_minus_malformed": "mean_minus_malformed",
}


FEATURES = [
    "selection_score",
    "proposal_exact",
    "mean_condition_selection_score",
    "min_condition_selection_score",
    "mean_exact_lift_vs_base",
    "min_exact_lift_vs_base",
    "max_malformed_regression_vs_base",
    "max_cap_hit_regression_vs_base",
    "sigma",
    "default_exact",
    "default_lift",
    "default_condition_score",
    "default_malformed",
    "default_output_tokens",
    "reordered_exact",
    "reordered_lift",
    "reordered_condition_score",
    "reordered_malformed",
    "reordered_output_tokens",
    "xml_exact",
    "xml_lift",
    "xml_condition_score",
    "xml_malformed",
    "xml_output_tokens",
    "valid_mean_exact",
    "valid_mean_lift",
    "valid_min_lift",
    "valid_max_lift",
    "valid_lift_spread",
    "all_mean_exact",
    "all_mean_lift",
    "all_min_lift",
    "all_max_lift",
    "malformed_cap_penalty",
]


def with_builtin_scores(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        row = dict(row)
        row["low_spread_valid_mean"] = row["valid_mean_lift"] - row["valid_lift_spread"]
        row["default_minus_instability"] = row["default_lift"] - row["valid_lift_spread"]
        row["mean_minus_malformed"] = row["valid_mean_lift"] - row["malformed_cap_penalty"]
        out.append(row)
    return out


def evaluate_selector(rows: list[dict[str, Any]], score_key: str, *, ks: list[int]) -> dict[str, Any]:
    rows = [row for row in rows if not np.isnan(float(row.get(score_key, np.nan)))]
    ordered = order_by(rows, score_key)
    dense_ranked = dense_order(rows)
    dense_best = dense_ranked[0] if dense_ranked else None
    dense_best_spec = None if dense_best is None else dense_best["spec"]
    dense_best_score = None if dense_best is None else float(dense_best["dense_exact"])
    selector_rank = {row["spec"]: idx + 1 for idx, row in enumerate(ordered)}
    scores = [float(row[score_key]) for row in rows]
    dense_scores = [float(row["dense_exact"]) for row in rows]
    result: dict[str, Any] = {
        "score_key": score_key,
        "candidates": len(rows),
        "dense_best_spec": dense_best_spec,
        "dense_best_score": dense_best_score,
        "dense_best_rank": None if dense_best_spec is None else selector_rank.get(dense_best_spec),
        "spearman": spearman(scores, dense_scores),
        "pearson": pearson(scores, dense_scores),
        "rows": [],
    }
    dense_sets = {k: {row["spec"] for row in dense_ranked[: min(k, len(dense_ranked))]} for k in ks}
    for k in ks:
        selected = ordered[: min(k, len(ordered))]
        selected_specs = {row["spec"] for row in selected}
        selected_best_dense = max((float(row["dense_exact"]) for row in selected), default=None)
        regret = None
        if dense_best_score is not None and selected_best_dense is not None:
            regret = dense_best_score - selected_best_dense
        result["rows"].append(
            {
                "k": k,
                "contains_dense_best": dense_best_spec in selected_specs if dense_best_spec else False,
                "dense_topk_overlap": len(selected_specs & dense_sets[k]),
                "selected_best_dense": selected_best_dense,
                "dense_regret": regret,
                "selected_specs": [row["spec"] for row in selected],
            }
        )
    return result


def train_linear(train_rows: list[dict[str, Any]], *, ridge: float = 1.0) -> dict[str, Any]:
    x = np.asarray([[float(row.get(key, 0.0)) for key in FEATURES] for row in train_rows], dtype=np.float64)
    y = np.asarray([float(row["dense_exact"]) for row in train_rows], dtype=np.float64)
    means = np.nanmean(x, axis=0)
    x = np.where(np.isnan(x), means, x)
    stds = np.std(x, axis=0)
    stds = np.where(stds < 1e-8, 1.0, stds)
    z = (x - means) / stds
    z = np.concatenate([np.ones((z.shape[0], 1)), z], axis=1)
    reg = np.eye(z.shape[1], dtype=np.float64) * ridge
    reg[0, 0] = 0.0
    weights = np.linalg.solve(z.T @ z + reg, z.T @ y)
    return {
        "features": FEATURES,
        "means": means.tolist(),
        "stds": stds.tolist(),
        "weights": weights.tolist(),
        "ridge": ridge,
    }


def apply_linear(rows: list[dict[str, Any]], model: dict[str, Any], score_key: str) -> list[dict[str, Any]]:
    means = np.asarray(model["means"], dtype=np.float64)
    stds = np.asarray(model["stds"], dtype=np.float64)
    weights = np.asarray(model["weights"], dtype=np.float64)
    out = []
    for row in rows:
        x = np.asarray([float(row.get(key, 0.0)) for key in model["features"]], dtype=np.float64)
        x = np.where(np.isnan(x), means, x)
        z = np.concatenate([[1.0], (x - means) / stds])
        scored = dict(row)
        scored[score_key] = float(z @ weights)
        out.append(scored)
    return out


def selector_rank_metric(result: dict[str, Any], *, k: int) -> tuple[float, float, float]:
    row = next((item for item in result["rows"] if int(item["k"]) == k), result["rows"][-1])
    regret = float("inf") if row["dense_regret"] is None else float(row["dense_regret"])
    contains = 0.0 if row["contains_dense_best"] else 1.0
    rank = float("inf") if result["dense_best_rank"] is None else float(result["dense_best_rank"])
    return (regret, contains, rank)


def analyze(runs: list[Path], *, ks: list[int], select_k: int) -> dict[str, Any]:
    panels = [RunPanel(panel.name, panel.root, with_builtin_scores(panel.rows)) for panel in [load_panel(run) for run in runs]]
    per_run: dict[str, Any] = {}
    for panel in panels:
        per_run[panel.name] = {
            "root": str(panel.root),
            "candidate_count": len(panel.rows),
            "selectors": {
                name: evaluate_selector(panel.rows, score_key, ks=ks) for name, score_key in SELECTORS.items()
            },
        }

    folds = []
    for test_idx, test in enumerate(panels):
        train = [panel for idx, panel in enumerate(panels) if idx != test_idx]
        train_rows = [row for panel in train for row in panel.rows]

        train_selector_results = {
            name: evaluate_selector(train_rows, score_key, ks=ks) for name, score_key in SELECTORS.items()
        }
        chosen_name, chosen_result = min(
            train_selector_results.items(),
            key=lambda item: selector_rank_metric(item[1], k=select_k),
        )
        chosen_score = SELECTORS[chosen_name]
        chosen_test = evaluate_selector(test.rows, chosen_score, ks=ks)

        linear = train_linear(train_rows)
        linear_rows = apply_linear(test.rows, linear, "linear_calibrated_score")
        linear_train_rows = apply_linear(train_rows, linear, "linear_calibrated_score")
        folds.append(
            {
                "train_runs": [panel.name for panel in train],
                "test_run": test.name,
                "chosen_fixed_selector": chosen_name,
                "chosen_fixed_train": chosen_result,
                "chosen_fixed_test": chosen_test,
                "linear_train": evaluate_selector(linear_train_rows, "linear_calibrated_score", ks=ks),
                "linear_test": evaluate_selector(linear_rows, "linear_calibrated_score", ks=ks),
                "linear_model": linear,
            }
        )

    summary = {
        "kind": "selector_calibration_audit",
        "runs": [str(run) for run in runs],
        "ks": ks,
        "select_k": select_k,
        "per_run": per_run,
        "folds": folds,
    }
    summary["verdict"] = verdict(summary)
    return summary


def fold_passes(result: dict[str, Any], *, k: int) -> bool:
    row = next((item for item in result["rows"] if int(item["k"]) == k), None)
    return bool(row and row["contains_dense_best"])


def verdict(summary: dict[str, Any]) -> dict[str, Any]:
    folds = summary["folds"]
    select_k = int(summary["select_k"])
    chosen_passes = [fold_passes(fold["chosen_fixed_test"], k=select_k) for fold in folds]
    linear_passes = [fold_passes(fold["linear_test"], k=select_k) for fold in folds]
    return {
        "fixed_selector_recovers_dense_best_all_folds": all(chosen_passes),
        "linear_recovers_dense_best_all_folds": all(linear_passes),
        "fixed_selector_pass_count": sum(chosen_passes),
        "linear_pass_count": sum(linear_passes),
        "fold_count": len(folds),
        "pass": all(chosen_passes) or all(linear_passes),
        "interpretation": (
            "A selector calibration is promising only if a rule chosen on training runs "
            "recovers the dense best within the configured k on every held-out run."
        ),
    }


def write_outputs(summary: dict[str, Any], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    rows_path = out / "selector_rows.csv"
    with rows_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scope",
                "run",
                "selector",
                "spearman",
                "pearson",
                "dense_best_rank",
                "k",
                "contains_dense_best",
                "dense_topk_overlap",
                "dense_regret",
            ],
        )
        writer.writeheader()
        for run_name, payload in summary["per_run"].items():
            for selector, result in payload["selectors"].items():
                for row in result["rows"]:
                    writer.writerow(flat_row("per_run", run_name, selector, result, row))
        for fold in summary["folds"]:
            for selector, result in [
                (f"chosen_fixed:{fold['chosen_fixed_selector']}", fold["chosen_fixed_test"]),
                ("linear_calibrated", fold["linear_test"]),
            ]:
                for row in result["rows"]:
                    writer.writerow(flat_row("heldout_fold", fold["test_run"], selector, result, row))

    (out / "report.md").write_text(render_report(summary))


def flat_row(scope: str, run: str, selector: str, result: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    return {
        "scope": scope,
        "run": run,
        "selector": selector,
        "spearman": result.get("spearman"),
        "pearson": result.get("pearson"),
        "dense_best_rank": result.get("dense_best_rank"),
        "k": row["k"],
        "contains_dense_best": row["contains_dense_best"],
        "dense_topk_overlap": row["dense_topk_overlap"],
        "dense_regret": row["dense_regret"],
    }


def fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def metric_at(result: dict[str, Any], k: int, key: str) -> Any:
    row = next((item for item in result["rows"] if int(item["k"]) == k), None)
    return None if row is None else row.get(key)


def render_report(summary: dict[str, Any]) -> str:
    select_k = int(summary["select_k"])
    lines = [
        "# Selector Calibration Audit",
        "",
        f"Runs: `{', '.join(summary['runs'])}`",
        "",
        "## Verdict",
        "",
        f"Gate: **{'PASS' if summary['verdict']['pass'] else 'FAIL'}**",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| fixed selector heldout pass count | {summary['verdict']['fixed_selector_pass_count']}/{summary['verdict']['fold_count']} |",
        f"| linear calibrated heldout pass count | {summary['verdict']['linear_pass_count']}/{summary['verdict']['fold_count']} |",
        "",
        "A pass means the selector chosen or trained on the other panel recovers the dense best "
        f"within top-{select_k} on the held-out panel.",
        "",
        "## Per-Run Selector Diagnostics",
        "",
    ]
    for run_name, payload in summary["per_run"].items():
        lines.extend([f"### {run_name}", "", "| selector | Spearman | dense best rank | top-8 contains best | top-8 regret |", "| --- | ---: | ---: | --- | ---: |"])
        ranked = sorted(
            payload["selectors"].items(),
            key=lambda item: selector_rank_metric(item[1], k=select_k),
        )
        for selector, result in ranked:
            lines.append(
                f"| {selector} | {fmt(result['spearman'])} | {fmt(result['dense_best_rank'])} | "
                f"{metric_at(result, select_k, 'contains_dense_best')} | {fmt(metric_at(result, select_k, 'dense_regret'))} |"
            )
        lines.append("")

    lines.extend(["## Held-Out Folds", "", "| train | test | selector | Spearman | dense best rank | top-8 contains best | top-8 regret |", "| --- | --- | --- | ---: | ---: | --- | ---: |"])
    for fold in summary["folds"]:
        train = ",".join(fold["train_runs"])
        for selector, result in [
            (f"chosen_fixed:{fold['chosen_fixed_selector']}", fold["chosen_fixed_test"]),
            ("linear_calibrated", fold["linear_test"]),
        ]:
            lines.append(
                f"| {train} | {fold['test_run']} | {selector} | {fmt(result['spearman'])} | "
                f"{fmt(result['dense_best_rank'])} | {metric_at(result, select_k, 'contains_dense_best')} | "
                f"{fmt(metric_at(result, select_k, 'dense_regret'))} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This audit is deliberately offline. It does not prove final LoRA quality because only the "
            "original robust shortlist has trusted PEFT confirmation. It tests the cheaper prerequisite: "
            "whether vLLM-derived scores can be calibrated to rank dense Gaussian screen winners across panels.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_ks(text: str) -> list[int]:
    return [int(item) for item in text.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit cross-panel vLLM selector calibration against dense PEFT screens.")
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ks", default="1,2,4,8,16,32")
    parser.add_argument("--select-k", type=int, default=8)
    args = parser.parse_args()
    summary = analyze(args.run, ks=parse_ks(args.ks), select_k=args.select_k)
    write_outputs(summary, args.out)
    print(json.dumps(summary["verdict"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
