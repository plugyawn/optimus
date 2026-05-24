from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


def as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def population_from_path(path: Path) -> int | None:
    for part in path.parts:
        match = re.fullmatch(r"p(\d+)", part)
        if match:
            return int(match.group(1))
        if part == "base":
            return 0
    return None


def flatten_metrics(payload: dict[str, Any], result_path: Path) -> list[dict[str, Any]]:
    population = population_from_path(result_path)
    model_name = str(payload.get("config_general", {}).get("model_name") or "")
    rows = []
    for task, metrics in (payload.get("results") or {}).items():
        if not isinstance(metrics, dict):
            continue
        for metric, value in metrics.items():
            number = as_number(value)
            if number is None:
                continue
            rows.append(
                {
                    "population": population,
                    "model": model_name,
                    "task": str(task),
                    "metric": str(metric),
                    "value": number,
                    "result_path": str(result_path),
                }
            )
    return rows


def result_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.glob("**/results_*.json")):
        try:
            rows.extend(flatten_metrics(json.loads(path.read_text()), path))
        except (json.JSONDecodeError, OSError):
            continue
    return sorted(rows, key=lambda row: (row["task"], row["metric"], row["population"] if row["population"] is not None else -1))


def add_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baselines = {
        (row["task"], row["metric"]): row["value"]
        for row in rows
        if row.get("population") == 0
    }
    out = []
    for row in rows:
        row = dict(row)
        base = baselines.get((row["task"], row["metric"]))
        row["base_value"] = base
        row["delta_vs_base"] = None if base is None else row["value"] - base
        out.append(row)
    return out


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")[:120] or "metric"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["population", "model", "task", "metric", "value", "base_value", "delta_vs_base", "result_path"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_plots(plot_dir: Path, rows: list[dict[str, Any]]) -> list[Path]:
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row["population"] is None:
            continue
        groups.setdefault((row["task"], row["metric"]), []).append(row)
    for (task, metric), group in sorted(groups.items()):
        group = sorted(group, key=lambda row: int(row["population"]))
        pops = [int(row["population"]) for row in group if int(row["population"]) > 0]
        values = [float(row["value"]) for row in group if int(row["population"]) > 0]
        if not pops:
            continue
        base = next((row.get("base_value") for row in group if row.get("base_value") is not None), None)
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        ax.plot(pops, values, marker="o", linewidth=2.0, color="#1f5f66", label="selected model")
        if base is not None:
            ax.axhline(float(base), color="#8f3f2f", linestyle="--", linewidth=1.5, label="base")
        ax.set_xscale("log", base=2)
        ax.set_xticks(pops)
        ax.set_xticklabels([str(pop) for pop in pops])
        ax.set_xlabel("search population")
        ax.set_ylabel(metric)
        ax.set_title(f"{task} / {metric}")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        stem = slug(f"{task}__{metric}")
        for suffix in [".png", ".pdf"]:
            path = plot_dir / f"{stem}{suffix}"
            fig.savefig(path, dpi=220 if suffix == ".png" else None)
            paths.append(path)
        plt.close(fig)
    return paths


def write_summary(path: Path, rows: list[dict[str, Any]], plots: list[Path]) -> None:
    best = {}
    for row in rows:
        if row["population"] in {None, 0}:
            continue
        key = (row["task"], row["metric"])
        if key not in best or row["value"] > best[key]["value"]:
            best[key] = row
    lines = ["# LightEval Population Report", ""]
    lines.append("| task | metric | best population | best value | base | delta |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for (task, metric), row in sorted(best.items()):
        base = "" if row.get("base_value") is None else f"{row['base_value']:.6g}"
        delta = "" if row.get("delta_vs_base") is None else f"{row['delta_vs_base']:.6g}"
        lines.append(f"| `{task}` | `{metric}` | {row['population']} | {row['value']:.6g} | {base} | {delta} |")
    lines.extend(["", "## Plots", ""])
    for plot in plots:
        if plot.suffix == ".png":
            lines.append(f"- `{plot}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize LightEval population outputs and generate publication-oriented plots.")
    parser.add_argument("--root", type=Path, default=Path("results/lighteval"))
    parser.add_argument("--out", type=Path, default=Path("results/lighteval/report"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = add_deltas(result_rows(args.root))
    if not rows:
        raise RuntimeError(f"no LightEval result JSON files found under {args.root}")
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "lighteval_metrics.csv", rows)
    plots = write_plots(args.out / "plots", rows)
    write_summary(args.out / "report.md", rows, plots)
    print(json.dumps({"rows": len(rows), "plots": [str(path) for path in plots], "out": str(args.out)}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
