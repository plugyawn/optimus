from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


PROTOCOL_KEYS = [
    "family",
    "population",
    "rank",
    "sigma",
    "targets",
    "screen_prompts",
    "max_new_tokens",
    "stop_at_answer",
    "antithetic",
]


@dataclass(frozen=True)
class ParsedCandidate:
    family: str
    seed: int
    sigma: float
    sign: int


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


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


def candidate_rows(run_dir: Path) -> list[dict]:
    rows = read_jsonl(run_dir / "candidate_summary.jsonl")
    if not rows:
        rows = read_jsonl(run_dir / "stage_candidate_summary.jsonl")
    if not rows:
        raise FileNotFoundError(f"no candidate summary jsonl found in {run_dir}")
    required = {"candidate", "exact_mean"}
    for idx, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise ValueError(f"{run_dir} candidate row {idx} missing columns: {sorted(missing)}")
    return rows


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for idx in order[i:j]:
            ranks[idx] = avg_rank
        i = j
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx = mean(xs)
    my = mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = sum((x - mx) ** 2 for x in xs) ** 0.5
    den_y = sum((y - my) ** 2 for y in ys) ** 0.5
    if den_x == 0.0 or den_y == 0.0:
        return None
    return num / (den_x * den_y)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    return pearson(average_ranks(xs), average_ranks(ys))


def top_candidates(rows: list[dict], score_col: str, k: int) -> set[str]:
    return {row["candidate"] for row in sorted(rows, key=lambda r: r[score_col], reverse=True)[:k]}


def compare_rankings(
    trusted_dir: Path,
    candidate_dir: Path,
    *,
    trusted_name: str,
    candidate_name: str,
    ks: list[int],
    spearman_gate: float,
    top8_gate: int,
) -> tuple[list[dict], dict]:
    trusted_rows = candidate_rows(trusted_dir)
    candidate_backend_rows = candidate_rows(candidate_dir)
    trusted_score = f"{trusted_name}_exact_mean"
    candidate_score = f"{candidate_name}_exact_mean"
    by_candidate = {
        row["candidate"]: {trusted_score: float(row["exact_mean"])}
        for row in trusted_rows
    }
    for row in candidate_backend_rows:
        if row["candidate"] in by_candidate:
            by_candidate[row["candidate"]][candidate_score] = float(row["exact_mean"])
    joined = [
        {"candidate": key, **value}
        for key, value in by_candidate.items()
        if candidate_score in value
    ]
    xs = [row[trusted_score] for row in joined]
    ys = [row[candidate_score] for row in joined]
    trusted_desc_ranks = average_ranks([-x for x in xs])
    candidate_desc_ranks = average_ranks([-y for y in ys])
    for row, trusted_rank, candidate_rank in zip(joined, trusted_desc_ranks, candidate_desc_ranks):
        row["abs_score_delta"] = abs(row[trusted_score] - row[candidate_score])
        row["trusted_rank"] = trusted_rank
        row["candidate_rank"] = candidate_rank
        row["rank_delta"] = abs(trusted_rank - candidate_rank)

    overlaps = {}
    for k in ks:
        k_eff = min(k, len(joined))
        overlaps[f"top{k}_overlap"] = len(
            top_candidates(joined, trusted_score, k_eff) & top_candidates(joined, candidate_score, k_eff)
        )
        overlaps[f"top{k}_possible"] = k_eff

    trusted_best = max(joined, key=lambda r: r[trusted_score]) if joined else {}
    candidate_best = max(joined, key=lambda r: r[candidate_score]) if joined else {}
    selected_regret = None
    if trusted_best and candidate_best:
        selected_regret = float(trusted_best[trusted_score] - candidate_best[trusted_score])
    top8_overlap = overlaps.get("top8_overlap")
    pass_top8 = top8_overlap is not None and top8_overlap >= min(top8_gate, overlaps.get("top8_possible", top8_gate))
    rho = spearman(xs, ys)
    summary = {
        "trusted_name": trusted_name,
        "candidate_name": candidate_name,
        "n_trusted": int(len(trusted_rows)),
        "n_candidate": int(len(candidate_backend_rows)),
        "n_common": int(len(joined)),
        "pearson": pearson(xs, ys),
        "spearman": rho,
        "mean_abs_score_delta": mean([row["abs_score_delta"] for row in joined]) if joined else None,
        "max_abs_score_delta": max([row["abs_score_delta"] for row in joined]) if joined else None,
        "trusted_best_candidate": trusted_best.get("candidate"),
        "trusted_best_score": trusted_best.get(trusted_score),
        "candidate_best_candidate": candidate_best.get("candidate"),
        "candidate_best_trusted_score": candidate_best.get(trusted_score),
        "candidate_best_score": candidate_best.get(candidate_score),
        "selected_regret_vs_trusted": selected_regret,
        "spearman_gate": spearman_gate,
        "top8_gate": top8_gate,
        "pass_spearman_gate": rho is not None and rho >= spearman_gate,
        "pass_top8_gate": pass_top8,
        **overlaps,
    }
    summary["pass"] = bool(summary["pass_spearman_gate"] and summary["pass_top8_gate"])
    return sorted(joined, key=lambda r: r[candidate_score], reverse=True), summary


def parse_ks(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def parse_candidate_key(key: str) -> ParsedCandidate:
    parts = key.split(":")
    if len(parts) != 4:
        raise ValueError(f"cannot parse candidate key: {key!r}")
    return ParsedCandidate(
        parts[0],
        int(parts[1].removeprefix("seed")),
        float(parts[2].removeprefix("s")),
        int(parts[3].removeprefix("sign")),
    )


def normalize(value):
    if isinstance(value, list):
        return ",".join(str(x) for x in value)
    return value


def protocol_checks(trusted_dir: Path, candidate_dir: Path, *, allow_missing_metadata: bool) -> tuple[list[dict], bool]:
    trusted = read_json(trusted_dir / "summary.json")
    candidate = read_json(candidate_dir / "summary.json")
    rows = []
    ok = True
    for key in PROTOCOL_KEYS:
        left = normalize(trusted.get(key))
        right = normalize(candidate.get(key))
        if left is None or right is None:
            passed = bool(allow_missing_metadata)
            note = "missing metadata"
        else:
            passed = left == right
            note = ""
        rows.append({"check": f"summary.{key}", "trusted": left, "candidate": right, "pass": passed, "note": note})
        ok = ok and passed
    for label, summary in [("trusted", trusted), ("candidate", candidate)]:
        overlap = summary.get("screen_holdout_overlap")
        passed = overlap == 0
        rows.append({"check": f"{label}.screen_holdout_overlap_zero", "trusted": overlap, "candidate": "", "pass": passed, "note": ""})
        ok = ok and passed
    return rows, ok


def rows_have_mode(path: Path, mode: str) -> bool:
    return any(row.get("mode") == mode for row in read_jsonl(path))


def base_row_checks(run_dir: Path, label: str) -> tuple[list[dict], bool]:
    per_prompt = run_dir / "per_prompt.jsonl"
    holdout = run_dir / "holdout_per_prompt.jsonl"
    rows = []
    ok = True
    for path, mode in [(per_prompt, "base_screen"), (holdout, "base_holdout")]:
        exists = path.exists()
        present = exists and rows_have_mode(path, mode)
        rows.append(
            {
                "check": f"{label}.{mode}_rows_present",
                "trusted": str(path),
                "candidate": "",
                "pass": present,
                "note": "" if exists else "missing file",
            }
        )
        ok = ok and present
    return rows, ok


def tensor_digest(tensor: Any) -> str:
    import torch

    data = bytes(tensor.detach().cpu().contiguous().view(torch.uint8).untyped_storage())
    return hashlib.sha256(data).hexdigest()


def resolve_adapter_model_path(candidate_dir: Path, spec: dict) -> Path:
    recorded_dir = Path(spec["path"])
    recorded_path = recorded_dir / "adapter_model.safetensors"
    try:
        if recorded_path.exists():
            return recorded_path
    except OSError:
        pass
    fallback_path = candidate_dir / "adapters" / recorded_dir.name / "adapter_model.safetensors"
    try:
        if fallback_path.exists():
            return fallback_path
    except OSError:
        pass
    return recorded_path


def check_adapter_tensors(
    candidate_dir: Path,
    *,
    model: str | None,
    rank: int | None,
    targets: list[str] | None,
    sample: int,
    local_files_only: bool,
) -> tuple[list[dict], dict]:
    adapters_path = candidate_dir / "adapters.jsonl"
    if not adapters_path.exists():
        return [], {"pass": False, "checked": 0, "reason": "missing adapters.jsonl"}
    if not model or rank is None or not targets:
        return [], {"pass": False, "checked": 0, "reason": "missing model/rank/targets metadata"}

    specs = read_jsonl(adapters_path)
    if sample > 0:
        specs = specs[:sample]
    rows = []
    ok = True
    checked = 0
    existing_specs = []
    for spec in specs:
        adapter_path = resolve_adapter_model_path(candidate_dir, spec)
        if not adapter_path.exists():
            rows.append(
                {
                    "adapter": spec.get("name"),
                    "candidate": spec.get("candidate"),
                    "module": "",
                    "tensor": "",
                    "pass": False,
                    "note": f"missing {adapter_path}",
                }
            )
            ok = False
            continue
        existing_specs.append((spec, adapter_path))
    if not existing_specs:
        return rows, {
            "pass": False,
            "checked": 0,
            "sampled_adapters": len(specs),
            "reason": "no sampled adapter files found",
        }

    import torch
    from safetensors.torch import load_file
    from transformers import AutoConfig

    from optimus.modeling import qwen_lora_shapes
    from optimus.modeling.noise import lora_noise_tensors

    config = AutoConfig.from_pretrained(model, trust_remote_code=True, local_files_only=local_files_only)
    shapes = qwen_lora_shapes(config, targets)
    family_state_path = candidate_dir / "family_state.pt"
    family_state = torch.load(family_state_path, map_location="cpu") if family_state_path.exists() else None
    for spec, adapter_path in existing_specs:
        candidate = parse_candidate_key(spec["candidate"])
        tensors = load_file(str(adapter_path))
        for module, in_features, out_features in shapes:
            a_key = f"base_model.model.{module}.lora_A.weight"
            b_key = f"base_model.model.{module}.lora_B.weight"
            expected_a, expected_b = lora_noise_tensors(
                module,
                (rank, in_features),
                (out_features, rank),
                candidate,
                rank,
                family_state=family_state,
                state_key=module,
            )
            for tensor_key, expected in [(a_key, expected_a), (b_key, expected_b)]:
                got = tensors.get(tensor_key)
                passed = got is not None and torch.equal(got.cpu(), expected.to(got.dtype))
                rows.append(
                    {
                        "adapter": spec.get("name"),
                        "candidate": spec.get("candidate"),
                        "module": module,
                        "tensor": tensor_key.rsplit(".", 2)[-2],
                        "pass": passed,
                        "expected_sha256": tensor_digest(expected.to(got.dtype)) if got is not None else "",
                        "actual_sha256": tensor_digest(got) if got is not None else "",
                        "note": "" if got is not None else "missing tensor",
                    }
                )
                ok = ok and passed
                checked += int(got is not None)
    reason = "" if checked > 0 else "no sampled adapter tensors checked"
    return rows, {"pass": ok and checked > 0, "checked": checked, "sampled_adapters": len(specs), "reason": reason}


def check_output_diff(
    path: Path | None,
    *,
    allow_missing: bool,
    max_exact_disagreement_rate: float,
    max_abs_exact_delta: float,
    max_abs_cap_hit_delta: float,
    max_abs_malformed_delta: float,
    min_answer_equal_rate: float,
) -> tuple[list[dict], dict]:
    if path is None or not path.exists():
        passed = bool(allow_missing)
        return [
            {
                "check": "output_diff_present",
                "trusted": str(path) if path else "",
                "candidate": "",
                "pass": passed,
                "note": "missing output diff summary",
            }
        ], {"pass": passed, "reason": "missing output diff summary"}
    payload = read_json(path)
    metric_specs = [
        ("exact_disagreement_rate", "<=", max_exact_disagreement_rate),
        ("max_abs_exact_delta_by_candidate", "<=", max_abs_exact_delta),
        ("max_abs_cap_hit_delta_by_candidate", "<=", max_abs_cap_hit_delta),
        ("max_abs_malformed_delta_by_candidate", "<=", max_abs_malformed_delta),
        ("answer_equal_rate", ">=", min_answer_equal_rate),
    ]
    rows = []
    ok = True
    for metric, op, threshold in metric_specs:
        value = payload.get(metric)
        if value is None:
            passed = False
            note = "missing metric"
        elif op == "<=":
            passed = float(value) <= threshold
            note = ""
        else:
            passed = float(value) >= threshold
            note = ""
        rows.append(
            {
                "check": f"output_diff.{metric}",
                "trusted": value,
                "candidate": threshold,
                "pass": passed,
                "note": note,
            }
        )
        ok = ok and passed
    return rows, {
        "pass": ok,
        "summary": str(path),
        "thresholds": {
            "max_exact_disagreement_rate": max_exact_disagreement_rate,
            "max_abs_exact_delta": max_abs_exact_delta,
            "max_abs_cap_hit_delta": max_abs_cap_hit_delta,
            "max_abs_malformed_delta": max_abs_malformed_delta,
            "min_answer_equal_rate": min_answer_equal_rate,
        },
        "metrics": {metric: payload.get(metric) for metric, _, _ in metric_specs},
    }


def write_markdown(path: Path, summary: dict, checks: list[dict]) -> None:
    status = "PASS" if summary["pass"] else "FAIL"
    lines = [
        "# Backend Parity Gate",
        "",
        f"Status: **{status}**",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        f"| protocol metadata | {summary['pass_protocol']} |",
        f"| base rows present | {summary['pass_base_rows']} |",
        f"| ranking correlation | {summary['pass_ranking']} |",
        f"| adapter tensor parity | {summary['pass_adapter_tensors']} |",
        f"| output diff parity | {summary['pass_output_diff']} |",
        "",
        "## Ranking",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| common candidates | {summary['ranking']['n_common']} |",
        f"| Spearman | {summary['ranking']['spearman']} |",
        f"| top8 overlap | {summary['ranking'].get('top8_overlap')}/{summary['ranking'].get('top8_possible')} |",
        f"| selected regret vs trusted | {summary['ranking']['selected_regret_vs_trusted']} |",
        "",
        "## Checks",
        "",
        "| check | pass | trusted | candidate | note |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for row in checks:
        lines.append(
            f"| {row.get('check', row.get('module', 'adapter_tensor'))} | {row.get('pass')} | "
            f"{row.get('trusted', row.get('adapter', ''))} | {row.get('candidate', '')} | {row.get('note', '')} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Strict gate for trusting vLLM candidate selection against a HF/PEFT reference.")
    p.add_argument("--trusted", required=True, type=Path)
    p.add_argument("--candidate", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--trusted-name", default="peft")
    p.add_argument("--candidate-name", default="vllm")
    p.add_argument("--ks", default="4,8,16")
    p.add_argument("--spearman-gate", type=float, default=0.85)
    p.add_argument("--top8-gate", type=int, default=6)
    p.add_argument("--adapter-sample", type=int, default=16)
    p.add_argument("--output-diff-summary", type=Path)
    p.add_argument("--max-exact-disagreement-rate", type=float, default=0.0)
    p.add_argument("--max-abs-exact-delta", type=float, default=0.0)
    p.add_argument("--max-abs-cap-hit-delta", type=float, default=0.0)
    p.add_argument("--max-abs-malformed-delta", type=float, default=0.0)
    p.add_argument("--min-answer-equal-rate", type=float, default=0.99)
    p.add_argument("--allow-missing-metadata", action="store_true")
    p.add_argument("--allow-missing-adapters", action="store_true")
    p.add_argument("--allow-missing-output-diff", action="store_true")
    p.add_argument("--local-files-only", action="store_true")
    args = p.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    protocol_rows, pass_protocol = protocol_checks(
        args.trusted,
        args.candidate,
        allow_missing_metadata=args.allow_missing_metadata,
    )
    trusted_base_rows, pass_trusted_base = base_row_checks(args.trusted, args.trusted_name)
    candidate_base_rows, pass_candidate_base = base_row_checks(args.candidate, args.candidate_name)
    joined, ranking = compare_rankings(
        args.trusted,
        args.candidate,
        trusted_name=args.trusted_name,
        candidate_name=args.candidate_name,
        ks=parse_ks(args.ks),
        spearman_gate=args.spearman_gate,
        top8_gate=args.top8_gate,
    )
    candidate_summary = read_json(args.candidate / "summary.json")
    adapter_rows, adapter_summary = check_adapter_tensors(
        args.candidate,
        model=candidate_summary.get("model"),
        rank=candidate_summary.get("rank"),
        targets=candidate_summary.get("targets"),
        sample=args.adapter_sample,
        local_files_only=args.local_files_only,
    )
    pass_adapters = bool(adapter_summary["pass"] or args.allow_missing_adapters)
    output_diff_rows, output_diff_summary = check_output_diff(
        args.output_diff_summary,
        allow_missing=args.allow_missing_output_diff,
        max_exact_disagreement_rate=args.max_exact_disagreement_rate,
        max_abs_exact_delta=args.max_abs_exact_delta,
        max_abs_cap_hit_delta=args.max_abs_cap_hit_delta,
        max_abs_malformed_delta=args.max_abs_malformed_delta,
        min_answer_equal_rate=args.min_answer_equal_rate,
    )
    pass_output_diff = bool(output_diff_summary["pass"])
    pass_base = bool(pass_trusted_base and pass_candidate_base)
    pass_ranking = bool(ranking["pass"])
    checks = protocol_rows + trusted_base_rows + candidate_base_rows
    if adapter_rows:
        checks.extend(adapter_rows[:256])
    else:
        checks.append(
            {
                "check": "adapter_tensor_parity",
                "trusted": "",
                "candidate": str(args.candidate),
                "pass": pass_adapters,
                "note": adapter_summary.get("reason", ""),
            }
        )
    checks.extend(output_diff_rows)
    summary = {
        "kind": "backend_parity_gate",
        "trusted": str(args.trusted),
        "candidate": str(args.candidate),
        "trusted_name": args.trusted_name,
        "candidate_name": args.candidate_name,
        "pass_protocol": pass_protocol,
        "pass_base_rows": pass_base,
        "pass_ranking": pass_ranking,
        "pass_adapter_tensors": pass_adapters,
        "pass_output_diff": pass_output_diff,
        "adapter_tensor_summary": adapter_summary,
        "output_diff_summary": output_diff_summary,
        "ranking": ranking,
    }
    summary["pass"] = bool(pass_protocol and pass_base and pass_ranking and pass_adapters and pass_output_diff)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_csv(args.out / "joined.csv", joined)
    write_csv(args.out / "checks.csv", checks)
    if adapter_rows:
        write_csv(args.out / "adapter_tensor_checks.csv", adapter_rows)
    write_markdown(args.out / "report.md", summary, checks)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
