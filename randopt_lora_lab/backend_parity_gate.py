from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .compare_backends import compare, parse_ks, read_jsonl, write_csv


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


def tensor_digest(tensor: torch.Tensor) -> str:
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

    from .lora_space import lora_noise_tensors
    from .vllm_lora_bench import qwen_lora_shapes

    config = AutoConfig.from_pretrained(model, trust_remote_code=True, local_files_only=local_files_only)
    shapes = qwen_lora_shapes(config, targets)
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
    joined, ranking = compare(
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
