from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from optimus.evaluation.validation import (
    SUBSPACE_JSON_PROVENANCE_FIELDS,
    SUBSPACE_SYSTEMS_AXIS_FIELDS,
    SUBSPACE_SYSTEMS_FIELDS,
    SUBSPACE_SYSTEMS_NUMERIC_FIELDS,
)


def as_float(value, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def is_json_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def timing_evidence_has_sync_marker(path: Path) -> bool:
    try:
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if isinstance(row, dict) and row.get("cuda_synchronized") is True:
                    return True
    except (OSError, json.JSONDecodeError):
        return False
    return False


def read_summary(path: Path) -> dict:
    row = json.loads(path.read_text())
    row["summary_path"] = str(path)
    row["run_dir"] = str(path.parent)
    row["suite"] = path.parts[-3] if len(path.parts) >= 3 else ""
    row["run"] = path.parent.name
    return row


def systems_summaries(root: Path) -> list[dict]:
    patterns = ["**/summary.json"]
    paths = []
    seen = set()
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if path in seen:
                continue
            seen.add(path)
            paths.append(path)
    return [read_summary(path) for path in paths]

def csv_write(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def jsonl_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def candidate_identity(row: dict) -> str:
    return str(row.get("candidate") or row.get("adapter") or row.get("name") or "")


def run_backend(row: dict) -> str:
    kind = str(row.get("kind", ""))
    if kind.startswith("vllm_"):
        return "vllm"
    if row.get("perturbation_backend") in {"dense", "lora"}:
        return "transformers"
    return str(row.get("backend") or "")


def run_method(row: dict) -> str:
    if row.get("method"):
        return str(row["method"])
    if row.get("perturbation_backend") in {"dense", "lora"}:
        return str(row["perturbation_backend"])
    if "lora" in str(row.get("kind", "")):
        return "lora"
    return ""


def is_search_summary(row: dict) -> bool:
    return row.get("kind") in {"vllm_lora_search", "search"}


def is_subspace_summary(row: dict) -> bool:
    return row.get("method") == "subspace" or str(row.get("kind", "")).startswith("subspace_")


def subspace_system_reports(rows: list[dict]) -> tuple[list[dict], list[str], list[str]]:
    reports: list[dict] = []
    missing: list[str] = []
    invalid: list[str] = []
    for row in rows:
        if not is_subspace_summary(row):
            continue
        path = Path(row["run_dir"]) / "systems_report.json"
        if not path.exists():
            missing.append(str(path))
            continue
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            invalid.append(f"{path}: invalid JSON")
            continue
        if payload.get("schema_version") != "subspace_systems_report_v1":
            invalid.append(f"{path}: invalid schema_version {payload.get('schema_version')!r}")
            continue
        absent = [
            field
            for field in (*SUBSPACE_JSON_PROVENANCE_FIELDS, *SUBSPACE_SYSTEMS_FIELDS, *SUBSPACE_SYSTEMS_AXIS_FIELDS)
            if field not in payload
        ]
        if absent:
            invalid.append(f"{path}: missing fields {', '.join(absent)}")
            continue
        bad_numeric = [field for field in SUBSPACE_SYSTEMS_NUMERIC_FIELDS if not is_json_number(payload.get(field))]
        if bad_numeric:
            invalid.append(f"{path}: nonnumeric fields {', '.join(bad_numeric)}")
            continue
        evidence_paths = payload.get("timing_evidence_paths")
        if not isinstance(evidence_paths, list) or not evidence_paths:
            invalid.append(f"{path}: missing timing_evidence_paths")
            continue
        missing_evidence = []
        for evidence in evidence_paths:
            evidence_path = Path(str(evidence))
            resolved = evidence_path if evidence_path.is_absolute() else path.parent / evidence_path
            if not resolved.exists():
                missing_evidence.append(str(evidence))
            elif not timing_evidence_has_sync_marker(resolved):
                missing_evidence.append(f"{evidence}:no_cuda_synchronized_marker")
        if missing_evidence:
            invalid.append(f"{path}: missing timing evidence {', '.join(missing_evidence)}")
            continue
        if payload.get("population") == 128:
            base_time = as_float(payload.get("base_model_time_s"), 0.0)
            qx_time = as_float(payload.get("qx_time_s"), -1.0)
            delta_time = as_float(payload.get("lazy_delta_time_s"), -1.0)
            if base_time <= 0.0 or qx_time < 0.0 or delta_time < 0.0:
                invalid.append(f"{path}: p128 speed gate missing base/qx/lazy timing")
                continue
            overhead = (qx_time + delta_time) / base_time
            if overhead > 0.25:
                invalid.append(f"{path}: p128 qx_plus_lazy_delta_overhead {overhead:.3f} > 0.25")
                continue
        if payload.get("prefix_cache_policy") != "disabled-for-search":
            invalid.append(f"{path}: invalid prefix_cache_policy {payload.get('prefix_cache_policy')!r}")
            continue
        enriched = payload | {
            "source_report": str(path),
            "source_run_dir": row["run_dir"],
            "population": payload.get("population", row.get("population")),
            "target_preset": payload.get("target_preset", row.get("target_preset")),
            "basis_rank": payload.get("basis_rank", row.get("basis_rank")),
            "kernel": payload.get("kernel", row.get("kernel")),
        }
        missing_axes = [field for field in SUBSPACE_SYSTEMS_AXIS_FIELDS if not enriched.get(field)]
        if missing_axes:
            invalid.append(f"{path}: missing comparison axes {', '.join(missing_axes)}")
            continue
        reports.append(enriched)
    return reports, missing, invalid


def selected_subspace_system_report(reports: list[dict]) -> dict:
    if not reports:
        return {}
    selected = min(reports, key=lambda row: as_float(row.get("candidates_per_sec"), float("inf")))
    return selected | {
        "systems_selection_policy": "slowest_candidates_per_sec_conservative_gate",
        "all_reports_count": len(reports),
        "all_report_sources": [row.get("source_report") for row in reports],
    }


def subspace_system_rows(reports: list[dict]) -> list[dict]:
    return sorted(
        [
            {
                "source_run_dir": row.get("source_run_dir"),
                "population": row.get("population"),
                "target_preset": row.get("target_preset"),
                "basis_rank": row.get("basis_rank"),
                "kernel": row.get("kernel"),
                "gpu_model": row.get("gpu_model"),
                "gpu_count": row.get("gpu_count"),
                "candidate_batch_size": row.get("candidate_batch_size"),
                "candidates_per_sec": row.get("candidates_per_sec"),
                "prompts_per_sec": row.get("prompts_per_sec"),
                "output_tokens_per_sec": row.get("output_tokens_per_sec"),
                "lazy_overhead_pct": row.get("lazy_overhead_pct"),
                "base_model_time_s": row.get("base_model_time_s"),
                "qx_time_s": row.get("qx_time_s"),
                "lazy_delta_time_s": row.get("lazy_delta_time_s"),
                "prefix_cache_policy": row.get("prefix_cache_policy"),
                "top_k_ensemble_cost_multiplier": row.get("top_k_ensemble_cost_multiplier"),
                "screen_score": row.get("screen_score"),
                "holdout_score": row.get("holdout_score"),
                "screen_to_holdout_drop": row.get("screen_to_holdout_drop"),
                "diversity_metrics": json.dumps(row.get("diversity_metrics", {}), sort_keys=True),
                "random_q_control": json.dumps(row.get("random_q_control", {}), sort_keys=True),
                "shuffled_q_control": json.dumps(row.get("shuffled_q_control", {}), sort_keys=True),
                "antithetic_odd_even": json.dumps(row.get("antithetic_odd_even", {}), sort_keys=True),
            }
            for row in reports
        ],
        key=lambda row: as_float(row.get("candidates_per_sec")),
        reverse=True,
    )


def append_subspace_systems_report(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("a") as f:
        f.write("\n## Subspace Systems\n\n")
        f.write(
            md_table(
                rows,
                [
                    "source_run_dir",
                    "gpu_model",
                    "population",
                    "target_preset",
                    "basis_rank",
                    "kernel",
                    "gpu_count",
                    "candidate_batch_size",
                    "candidates_per_sec",
                    "prompts_per_sec",
                    "output_tokens_per_sec",
                    "lazy_overhead_pct",
                    "top_k_ensemble_cost_multiplier",
                    "screen_score",
                    "holdout_score",
                    "screen_to_holdout_drop",
                    "prefix_cache_policy",
                ],
            )
        )
        f.write("\n")


def write_subspace_fail_closed_report(path: Path, *, missing: list[str], invalid: list[str]) -> None:
    lines = [
        "# Optimus Systems Report",
        "",
        "Subspace systems reporting failed closed. Every subspace search summary must have a measured",
        "`systems_report.json` with the required subspace runtime fields before the suite-level report can pass.",
        "",
    ]
    if missing:
        lines.extend(["## Missing", "", *[f"- `{item}`" for item in missing], ""])
    if invalid:
        lines.extend(["## Invalid", "", *[f"- {item}" for item in invalid], ""])
    path.write_text("\n".join(lines))


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
        if not is_search_summary(row):
            continue
        out.append(
            {
                "suite": row["suite"],
                "run": row["run"],
                "backend": run_backend(row),
                "method": run_method(row),
                "run_dir": row["run_dir"],
                "summary_path": row["summary_path"],
                "population": row.get("population"),
                "screen_prompts": row.get("screen_prompts"),
                "chunk_adapters": row.get("chunk_adapters"),
                "tensor_parallel_size": row.get("tensor_parallel_size"),
                "max_loras": row.get("max_loras"),
                "max_new_tokens": row.get("max_new_tokens"),
                "enforce_eager": row.get("enforce_eager"),
                "max_num_batched_tokens": row.get("max_num_batched_tokens"),
                "candidate_sec": row.get("candidate_sec"),
                "screen_prompts_per_sec": row.get("screen_prompts_per_sec") or row.get("pair_sec"),
                "screen_tokens_per_sec": row.get("screen_tokens_per_sec"),
                "holdout_tokens_per_sec": row.get("holdout_tokens_per_sec"),
                "best_tokens_per_sec": row.get("best_tokens_per_sec"),
                "eval_elapsed_s": row.get("eval_elapsed_s"),
                "load_s": row.get("load_s"),
            }
        )
    return sorted(out, key=lambda r: (r.get("candidate_sec") or 0), reverse=True)


def bench_rows(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        if row.get("kind") != "vllm_lora_bench":
            continue
        out.append(
            {
                "suite": row["suite"],
                "run": row["run"],
                "backend": run_backend(row),
                "method": run_method(row),
                "adapters": row.get("adapters"),
                "prompts": row.get("prompts"),
                "tensor_parallel_size": row.get("tensor_parallel_size"),
                "max_new_tokens": row.get("max_new_tokens"),
                "max_loras": row.get("max_loras"),
                "preload": row.get("preload"),
                "lora_tokens_per_sec": row.get("lora_tokens_per_sec"),
                "lora_prompts_per_sec": row.get("lora_prompts_per_sec"),
                "best_adapter_tokens_per_sec": row.get("best_adapter_tokens_per_sec"),
                "best_adapter_prompts_per_sec": row.get("best_adapter_prompts_per_sec"),
                "base_tokens_per_sec": row.get("base_tokens_per_sec"),
                "mixed_tokens_per_sec": row.get("mixed_tokens_per_sec"),
                "mixed_prompts_per_sec": row.get("mixed_prompts_per_sec"),
                "load_s": row.get("load_s"),
                "preload_s": row.get("preload_s"),
            }
        )
    return sorted(out, key=lambda r: (as_int(r.get("adapters")), str(r.get("suite")), str(r.get("run"))))


def quality_rows(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        if not is_search_summary(row):
            continue
        top_screen = row.get("top_screen") or []
        top_holdout = row.get("top_holdout") or []
        screen_selected = top_screen[0] if top_screen else {}
        screen_selected_key = candidate_identity(screen_selected)
        holdout_by_candidate = {candidate_identity(item): item for item in top_holdout}
        selected_holdout = holdout_by_candidate.get(screen_selected_key)
        holdout_oracle = max(top_holdout, key=lambda item: as_float(item.get("exact_mean")), default={})
        screen_selected_exact = None if not screen_selected else as_float(screen_selected.get("exact_mean"))
        screen_selected_holdout_exact = None if selected_holdout is None else as_float(selected_holdout.get("exact_mean"))
        promoted_holdout_oracle_exact = None if not holdout_oracle else as_float(holdout_oracle.get("exact_mean"))
        base_screen_exact = row.get("base_screen_exact")
        base_holdout_exact = row.get("base_holdout_exact")
        out.append(
            {
                "suite": row["suite"],
                "run": row["run"],
                "backend": run_backend(row),
                "method": run_method(row),
                "population": row.get("population"),
                "screen_prompts": row.get("screen_prompts"),
                "holdout_prompts": row.get("holdout_prompts"),
                "base_screen_exact": base_screen_exact,
                "base_holdout_exact": base_holdout_exact,
                "screen_selected_candidate": screen_selected_key,
                "screen_selected_exact": screen_selected_exact,
                "screen_selected_holdout_exact": screen_selected_holdout_exact,
                "promoted_holdout_oracle_candidate": candidate_identity(holdout_oracle),
                "promoted_holdout_oracle_exact": promoted_holdout_oracle_exact,
                "screen_selected_delta_vs_base": None
                if screen_selected_exact is None
                else screen_selected_exact - as_float(base_screen_exact),
                "screen_selected_holdout_delta_vs_base": None
                if screen_selected_holdout_exact is None
                else screen_selected_holdout_exact - as_float(base_holdout_exact),
                "promoted_holdout_oracle_delta_vs_base": None
                if promoted_holdout_oracle_exact is None
                else promoted_holdout_oracle_exact - as_float(base_holdout_exact),
                "best_ensemble_holdout_exact": row.get("best_ensemble_holdout_exact"),
                "best_strict_ensemble_holdout_exact": row.get("best_strict_ensemble_holdout_exact"),
                "candidate_sec": row.get("candidate_sec"),
                "screen_tokens_per_sec": row.get("screen_tokens_per_sec"),
                "holdout_tokens_per_sec": row.get("holdout_tokens_per_sec"),
            }
        )
    return sorted(out, key=lambda r: (as_int(r.get("population")), str(r.get("suite")), str(r.get("run"))))


def best_of_n_rows(full: list[dict]) -> list[dict]:
    out = []
    for row in sorted(full, key=lambda r: (str(r.get("suite")), str(r.get("run")))):
        candidates = jsonl_rows(Path(row["run_dir"]) / "candidate_summary.jsonl")
        if not candidates:
            continue
        candidates = sorted(candidates, key=lambda item: as_int(item.get("adapter_index")))
        base = as_float(row.get("base_screen_exact"), 0.0)
        running_best = float("-inf")
        running_candidate = ""
        for idx, candidate in enumerate(candidates, start=1):
            score = as_float(candidate.get("exact_mean"), float("-inf"))
            if score > running_best:
                running_best = score
                running_candidate = str(candidate.get("candidate") or candidate.get("adapter") or "")
            out.append(
                {
                    "suite": row["suite"],
                    "run": row["run"],
                    "backend": row.get("backend"),
                    "method": row.get("method"),
                    "population": row.get("population"),
                    "screen_prompts": row.get("screen_prompts"),
                    "n": idx,
                    "best_screen_exact": running_best,
                    "base_screen_exact": base,
                    "delta_vs_base": running_best - base,
                    "best_candidate": running_candidate,
                }
            )
    return out


def parity_rows(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        if row.get("kind") != "backend_parity_gate":
            continue
        ranking = row.get("ranking") or {}
        out.append(
            {
                "suite": row["suite"],
                "run": row["run"],
                "trusted_name": row.get("trusted_name") or ranking.get("trusted_name"),
                "candidate_name": row.get("candidate_name") or ranking.get("candidate_name"),
                "n_common": ranking.get("n_common"),
                "spearman": ranking.get("spearman"),
                "top8_overlap": ranking.get("top8_overlap"),
                "top8_possible": ranking.get("top8_possible"),
                "selected_regret_vs_trusted": ranking.get("selected_regret_vs_trusted"),
                "pass": row.get("pass"),
                "pass_protocol": row.get("pass_protocol"),
                "pass_base_rows": row.get("pass_base_rows"),
                "pass_adapter_tensors": row.get("pass_adapter_tensors"),
                "pass_output_diff": row.get("pass_output_diff"),
                "output_diff_reason": (row.get("output_diff_summary") or {}).get("reason", ""),
                "trusted_best_candidate": ranking.get("trusted_best_candidate"),
                "candidate_best_candidate": ranking.get("candidate_best_candidate"),
            }
        )
    return sorted(out, key=lambda r: (str(r.get("suite")), str(r.get("run"))))


def halving_rows(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        if row.get("kind") not in {"halving_recall", "vllm_lora_halving", "halving"}:
            continue
        top_stage = row.get("top_stage") or []
        top_screen = row.get("top_screen") or []
        top_holdout = row.get("top_holdout") or []
        stage_best = top_stage[0] if top_stage else {}
        screen_best = top_screen[0] if top_screen else {}
        holdout_by_candidate = {candidate_identity(item): item for item in top_holdout}
        selected_holdout = holdout_by_candidate.get(candidate_identity(screen_best), {})
        out.append(
            {
                "suite": row["suite"],
                "run": row["run"],
                "backend": run_backend(row),
                "method": run_method(row),
                "screen_prompts": row.get("screen_prompts"),
                "stage_prompts": row.get("stage_prompts"),
                "survivors": row.get("survivors"),
                "candidate_sec": row.get("candidate_sec"),
                "prompt_eval_savings": row.get("prompt_eval_savings"),
                "top8_survivor_recall": row.get("top8_survivor_recall"),
                "top8_possible": row.get("top8_possible"),
                "full_best_survived": row.get("full_best_survived"),
                "halving_selected_regret_vs_full": row.get("halving_selected_regret_vs_full"),
                "stage_selected_candidate": candidate_identity(stage_best),
                "stage_selected_exact": stage_best.get("exact_mean"),
                "screen_selected_candidate": candidate_identity(screen_best),
                "screen_selected_exact": screen_best.get("exact_mean"),
                "screen_selected_holdout_exact": selected_holdout.get("exact_mean"),
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
        if row.get("population") in {512, 1024, 2048, 4096}
        and row.get("max_new_tokens") in {16, 32}
        and row.get("screen_prompts") in {64, 128}
    ][:18]
    labels = [f"{row['suite']}/{row['run'].replace('search_', '')}" for row in selected]
    values = [row.get("candidate_sec") or 0.0 for row in selected]
    fig, ax = plt.subplots(figsize=(11, max(4, 0.38 * len(selected))))
    ax.barh(range(len(selected)), values, color="#2f6f73")
    ax.set_yticks(range(len(selected)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("candidates/sec")
    ax.set_title("Optimus full-search throughput")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_adapter_throughput(path: Path, rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = rows[:18]
    labels = [f"a{row.get('adapters')}/p{row.get('prompts')}/tp{row.get('tensor_parallel_size') or 1}" for row in selected]
    lora_values = [as_float(row.get("lora_tokens_per_sec")) for row in selected]
    mixed_values = [as_float(row.get("mixed_tokens_per_sec")) for row in selected]
    fig, ax = plt.subplots(figsize=(11, max(4, 0.42 * len(selected))))
    y = list(range(len(selected)))
    ax.barh([idx - 0.18 for idx in y], lora_values, height=0.35, color="#2f6f73", label="sequential LoRA")
    ax.barh([idx + 0.18 for idx in y], mixed_values, height=0.35, color="#7a5c2e", label="mixed LoRA batch")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("tokens/sec")
    ax.set_title("Adapter-throughput scaling")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_token_throughput(path: Path, rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [
        row
        for row in rows
        if row.get("population") in {512, 1024, 2048, 4096}
        and row.get("max_new_tokens") in {16, 32}
        and row.get("screen_prompts") in {64, 128}
        and (row.get("screen_tokens_per_sec") is not None or row.get("best_tokens_per_sec") is not None)
    ][:18]
    labels = [f"{row['suite']}/{row['run'].replace('search_', '')}/tp{row.get('tensor_parallel_size') or 1}" for row in selected]
    screen_values = [row.get("screen_tokens_per_sec") or 0.0 for row in selected]
    best_values = [row.get("best_tokens_per_sec") or 0.0 for row in selected]
    fig, ax = plt.subplots(figsize=(11, max(4, 0.42 * len(selected))))
    y = list(range(len(selected)))
    ax.barh([idx - 0.18 for idx in y], screen_values, height=0.35, color="#2f6f73", label="screen tokens/sec")
    ax.barh([idx + 0.18 for idx in y], best_values, height=0.35, color="#7a5c2e", label="best tokens/sec")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("tokens/sec")
    ax.set_title("Optimus token-throughput scaling")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_best_of_n(path: Path, rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_run: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        by_run.setdefault((str(row["suite"]), str(row["run"])), []).append(row)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for (suite, run), run_rows in sorted(by_run.items()):
        xs = [as_int(row.get("n")) for row in run_rows]
        ys = [as_float(row.get("best_screen_exact")) for row in run_rows]
        if not xs:
            continue
        label = f"{suite}/{run}".replace("optimus_gpu_suite/", "")
        ax.plot(xs, ys, linewidth=1.7, label=label)
    ax.set_xlabel("candidate evaluations")
    ax.set_ylabel("running best screen exact")
    ax.set_title("Best-of-N search curve")
    ax.set_xscale("log", base=2)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_quality_scaling(path: Path, rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [row for row in rows if row.get("population")]
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = [as_int(row.get("population")) for row in selected]
    screen = [as_float(row.get("screen_selected_exact")) for row in selected]
    selected_holdout = [as_float(row.get("screen_selected_holdout_exact")) for row in selected]
    holdout_oracle = [as_float(row.get("promoted_holdout_oracle_exact")) for row in selected]
    ensemble = [as_float(row.get("best_ensemble_holdout_exact")) for row in selected]
    base_holdout = [as_float(row.get("base_holdout_exact")) for row in selected]
    if xs:
        ax.plot(xs, screen, marker="o", label="screen-selected screen")
        ax.plot(xs, selected_holdout, marker="o", label="screen-selected holdout")
        ax.plot(xs, holdout_oracle, marker="o", label="promoted holdout oracle")
        ax.plot(xs, ensemble, marker="o", label="best ensemble holdout")
        ax.plot(xs, base_holdout, linestyle="--", color="#777777", label="base holdout")
    ax.set_xlabel("population")
    ax.set_ylabel("exact accuracy")
    ax.set_title("Search-quality scaling")
    if xs and min(xs) > 0:
        ax.set_xscale("log", base=2)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
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
        if row.get("prompt_eval_savings") is None or row.get("halving_selected_regret_vs_full") is None:
            continue
        savings = row.get("prompt_eval_savings")
        regret = row.get("halving_selected_regret_vs_full")
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


def write_report(
    path: Path,
    full: list[dict],
    bench: list[dict],
    quality: list[dict],
    best_of_n: list[dict],
    parity: list[dict],
    halving: list[dict],
) -> None:
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
    transfer_failures = [
        row
        for row in quality
        if row.get("screen_selected_holdout_delta_vs_base") is not None
        and row.get("screen_selected_holdout_delta_vs_base") < 0
    ]
    fastest_raw_line = "- No full-search rows found."
    if fastest_raw:
        fastest_raw_line = (
            f"- Fastest raw full-search row: `{fastest_raw.get('suite')}/{fastest_raw.get('run')}` "
            f"at `{fastest_raw.get('candidate_sec'):.4g}` candidates/sec."
        )
    fastest_matched_line = "- No matched-protocol full-search rows found."
    if fastest_matched:
        fastest_matched_line = (
            f"- Fastest matched-protocol full search: `{fastest_matched.get('suite')}/{fastest_matched.get('run')}` "
            f"at `{fastest_matched.get('candidate_sec'):.4g}` candidates/sec."
        )
    selection_line = "- Quality rows separate screen-selected heldout transfer from promoted holdout-oracle quality."
    if transfer_failures:
        failed = ", ".join(f"`{row.get('suite')}/{row.get('run')}`" for row in transfer_failures[:4])
        selection_line = (
            f"- Screen-selected heldout transfer regressed versus base for {failed}; "
            "use the promoted holdout-oracle column only as candidate-generation evidence."
        )
    plot_names = []
    if full:
        plot_names.extend(["full_search_candidate_sec.png", "token_throughput.png"])
    if bench:
        plot_names.append("adapter_throughput.png")
    if best_of_n:
        plot_names.append("best_of_n.png")
    if quality:
        plot_names.append("quality_scaling.png")
    if parity:
        plot_names.append("parity_gates.png")
    if halving:
        plot_names.append("halving_tradeoff.png")
    plot_line = "Plots: " + ", ".join(f"`{name}`" for name in plot_names) + "." if plot_names else "Plots: none."
    lines = [
        "# Optimus Systems Report",
        "",
        "## Executive Call",
        "",
        fastest_raw_line,
        fastest_matched_line,
        selection_line,
        "- Staged search, when present, is judged by selected regret, full-best survival, and top-k survivor recall on matched full-search panels.",
        "",
        "## Full Search",
        "",
        md_table(
            full,
            [
                "suite",
                "run",
                "backend",
                "method",
                "population",
                "screen_prompts",
                "chunk_adapters",
                "tensor_parallel_size",
                "max_new_tokens",
                "candidate_sec",
                "screen_prompts_per_sec",
                "screen_tokens_per_sec",
                "best_tokens_per_sec",
                "eval_elapsed_s",
            ],
            limit=14,
        ),
        "",
        "## Adapter Throughput",
        "",
        md_table(
            bench,
            [
                "suite",
                "run",
                "backend",
                "method",
                "adapters",
                "prompts",
                "tensor_parallel_size",
                "lora_tokens_per_sec",
                "mixed_tokens_per_sec",
                "lora_prompts_per_sec",
                "mixed_prompts_per_sec",
                "load_s",
            ],
        ),
        "",
        "## Quality Scaling",
        "",
        md_table(
            quality,
            [
                "suite",
                "run",
                "backend",
                "method",
                "population",
                "base_holdout_exact",
                "screen_selected_holdout_exact",
                "screen_selected_holdout_delta_vs_base",
                "promoted_holdout_oracle_exact",
                "promoted_holdout_oracle_delta_vs_base",
                "best_ensemble_holdout_exact",
                "best_strict_ensemble_holdout_exact",
            ],
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
        f"- Best-of-N points: `{len(best_of_n)}`.",
        "",
        plot_line,
        "",
    ]
    path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a focused Optimus systems report.")
    parser.add_argument("--root", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    rows = systems_summaries(args.root)
    subspace_rows = [row for row in rows if is_subspace_summary(row)]
    subspace_reports, missing_subspace_reports, invalid_subspace_reports = subspace_system_reports(rows)
    if subspace_rows:
        if missing_subspace_reports or invalid_subspace_reports:
            write_subspace_fail_closed_report(
                args.out / "report.md",
                missing=missing_subspace_reports,
                invalid=invalid_subspace_reports,
            )
            return 1
        selected_subspace = selected_subspace_system_report(subspace_reports)
        if not selected_subspace:
            write_subspace_fail_closed_report(
                args.out / "report.md",
                missing=["no valid subspace systems_report.json files found"],
                invalid=[],
            )
            return 1
        (args.out / "systems_report.json").write_text(json.dumps(selected_subspace, indent=2, sort_keys=True) + "\n")
        csv_write(
            args.out / "subspace_systems.csv",
            subspace_system_rows(subspace_reports),
            [
                "source_run_dir",
                "gpu_model",
                "population",
                "target_preset",
                "basis_rank",
                "kernel",
                "gpu_count",
                "candidate_batch_size",
                "candidates_per_sec",
                "prompts_per_sec",
                "output_tokens_per_sec",
                "lazy_overhead_pct",
                "base_model_time_s",
                "qx_time_s",
                "lazy_delta_time_s",
                "prefix_cache_policy",
                "top_k_ensemble_cost_multiplier",
                "screen_score",
                "holdout_score",
                "screen_to_holdout_drop",
                "diversity_metrics",
                "random_q_control",
                "shuffled_q_control",
                "antithetic_odd_even",
            ],
        )
    full = full_search_rows(rows)
    bench = bench_rows(rows)
    quality = quality_rows(rows)
    best_of_n = best_of_n_rows(full)
    parity = parity_rows(rows)
    halving = halving_rows(rows)
    csv_write(
        args.out / "full_search.csv",
        full,
        [
            "suite",
            "run",
            "backend",
            "method",
            "population",
            "screen_prompts",
            "chunk_adapters",
            "tensor_parallel_size",
            "max_loras",
            "max_new_tokens",
            "enforce_eager",
            "max_num_batched_tokens",
            "candidate_sec",
            "screen_prompts_per_sec",
            "screen_tokens_per_sec",
            "holdout_tokens_per_sec",
            "best_tokens_per_sec",
            "eval_elapsed_s",
            "load_s",
        ],
    )
    csv_write(
        args.out / "bench.csv",
        bench,
        [
            "suite",
            "run",
            "backend",
            "method",
            "adapters",
            "prompts",
            "tensor_parallel_size",
            "max_new_tokens",
            "max_loras",
            "preload",
            "lora_tokens_per_sec",
            "lora_prompts_per_sec",
            "best_adapter_tokens_per_sec",
            "best_adapter_prompts_per_sec",
            "base_tokens_per_sec",
            "mixed_tokens_per_sec",
            "mixed_prompts_per_sec",
            "load_s",
            "preload_s",
        ],
    )
    csv_write(
        args.out / "quality_scaling.csv",
        quality,
        [
            "suite",
            "run",
            "backend",
            "method",
            "population",
            "screen_prompts",
            "holdout_prompts",
            "base_screen_exact",
            "base_holdout_exact",
            "screen_selected_candidate",
            "screen_selected_exact",
            "screen_selected_holdout_exact",
            "promoted_holdout_oracle_candidate",
            "promoted_holdout_oracle_exact",
            "screen_selected_delta_vs_base",
            "screen_selected_holdout_delta_vs_base",
            "promoted_holdout_oracle_delta_vs_base",
            "best_ensemble_holdout_exact",
            "best_strict_ensemble_holdout_exact",
            "candidate_sec",
            "screen_tokens_per_sec",
            "holdout_tokens_per_sec",
        ],
    )
    csv_write(
        args.out / "best_of_n.csv",
        best_of_n,
        [
            "suite",
            "run",
            "backend",
            "method",
            "population",
            "screen_prompts",
            "n",
            "best_screen_exact",
            "base_screen_exact",
            "delta_vs_base",
            "best_candidate",
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
            "pass_protocol",
            "pass_base_rows",
            "pass_adapter_tensors",
            "pass_output_diff",
            "output_diff_reason",
        ],
    )
    csv_write(
        args.out / "halving.csv",
        halving,
        [
            "suite",
            "run",
            "backend",
            "method",
            "screen_prompts",
            "stage_prompts",
            "survivors",
            "candidate_sec",
            "prompt_eval_savings",
            "top8_survivor_recall",
            "top8_possible",
            "full_best_survived",
            "halving_selected_regret_vs_full",
            "stage_selected_candidate",
            "stage_selected_exact",
            "screen_selected_candidate",
            "screen_selected_exact",
            "screen_selected_holdout_exact",
            "eval_elapsed_s",
        ],
    )
    if full:
        plot_full_search(args.out / "full_search_candidate_sec.png", full)
        plot_token_throughput(args.out / "token_throughput.png", full)
    if bench:
        plot_adapter_throughput(args.out / "adapter_throughput.png", bench)
    if best_of_n:
        plot_best_of_n(args.out / "best_of_n.png", best_of_n)
    if quality:
        plot_quality_scaling(args.out / "quality_scaling.png", quality)
    if parity:
        plot_parity(args.out / "parity_gates.png", parity)
    if halving:
        plot_halving(args.out / "halving_tradeoff.png", halving)
    write_report(args.out / "report.md", full, bench, quality, best_of_n, parity, halving)
    append_subspace_systems_report(args.out / "report.md", subspace_system_rows(subspace_reports))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
