from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ARTIFACTS = [
    ("dense_summary", "dense/summary.json", ("preflight", "confirm"), "summary"),
    ("vllm_summary", "vllm/summary.json", ("preflight", "confirm"), "summary"),
    ("shortlist", "shortlist_top*.jsonl", ("preflight", "confirm"), "jsonl"),
    ("preflight", "preflight_summary.json", ("preflight", "confirm"), "pass"),
    ("score_sanity", "score_sanity/summary.json", ("preflight", "confirm"), "pass"),
    ("confirmed_summary", "confirmed/summary.json", ("confirm",), "summary"),
    ("confirmed_validity", "confirmed/validity/summary.json", ("confirm",), "pass"),
    ("shortlist_dense_confirmation", "shortlist_dense_confirmation/summary.json", ("confirm",), "gate"),
    ("family_state_provenance", "family_state_provenance_audit/summary.json", ("confirm",), "pass"),
    ("search_quality_confirmation", "search_quality_confirmation/summary.json", ("confirm",), "gate"),
    ("current_goal_audit", "current_goal_audit/summary.json", ("confirm",), "pass"),
]


def read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, None
    try:
        return json.loads(path.read_text()), None
    except json.JSONDecodeError as exc:
        return None, f"{exc.__class__.__name__}: {exc}"


def count_jsonl(path: Path) -> int:
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def gate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    gate = payload.get("gate")
    return gate if isinstance(gate, dict) else payload


def extract_pass(payload: dict[str, Any]) -> bool | None:
    value = payload.get("pass")
    return value if isinstance(value, bool) else None


def extract_failed(payload: dict[str, Any]) -> list[Any]:
    failed = payload.get("failed", [])
    return failed if isinstance(failed, list) else [failed]


def resolve_artifact_path(root: Path, relpath: str) -> Path | None:
    if "*" not in relpath:
        return root / relpath
    matches = sorted(root.glob(relpath))
    return matches[0] if matches else None


def artifact_row(root: Path, name: str, relpath: str, required_modes: tuple[str, ...], pass_source: str) -> dict[str, Any]:
    path = resolve_artifact_path(root, relpath)
    row: dict[str, Any] = {
        "name": name,
        "path": str(path) if path else None,
        "relpath": str(path.relative_to(root)) if path else relpath,
        "present": bool(path and path.exists()),
        "required_modes": list(required_modes),
        "pass": None,
        "failed": [],
        "kind": None,
        "rows": None,
        "readable": None,
        "error": None,
    }
    if not (path and path.exists()):
        return row
    if pass_source == "jsonl":
        row["rows"] = count_jsonl(path)
        row["readable"] = True
        return row
    payload, error = read_json(path)
    if payload is None:
        row["readable"] = False
        row["error"] = error
        return row
    row["readable"] = True
    row["kind"] = payload.get("kind")
    if pass_source == "pass":
        row["pass"] = extract_pass(payload)
        row["failed"] = extract_failed(payload)
    elif pass_source == "gate":
        gate = gate_payload(payload)
        row["pass"] = extract_pass(gate)
        row["failed"] = extract_failed(gate)
    return row


def infer_mode(root: Path) -> str:
    return "confirm" if (root / "confirmed" / "summary.json").exists() else "preflight"


def build_manifest(root: Path, *, mode: str = "auto") -> dict[str, Any]:
    resolved_mode = infer_mode(root) if mode == "auto" else mode
    if resolved_mode not in {"preflight", "confirm"}:
        raise ValueError(f"mode must be preflight, confirm, or auto; got {mode!r}")
    artifacts = [artifact_row(root, *spec) for spec in ARTIFACTS]
    required = [row for row in artifacts if resolved_mode in row["required_modes"]]
    missing_required = [row["name"] for row in required if not row["present"]]
    unreadable_required = [row["name"] for row in required if row["readable"] is False]
    failed_gates = [row["name"] for row in required if row["pass"] is False]
    goal = next((row for row in artifacts if row["name"] == "current_goal_audit"), None)
    return {
        "kind": "qproj_replay_manifest",
        "root": str(root),
        "mode": resolved_mode,
        "artifact_complete": not missing_required and not unreadable_required,
        "method_pass": bool(goal and goal["pass"] is True),
        "missing_required": missing_required,
        "unreadable_required": unreadable_required,
        "failed_gates": failed_gates,
        "artifacts": artifacts,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Replay Manifest",
        "",
        f"Root: `{summary['root']}`",
        f"Mode: `{summary['mode']}`",
        f"Artifact complete: `{str(summary['artifact_complete']).lower()}`",
        f"Method pass: `{str(summary['method_pass']).lower()}`",
        "",
        "| artifact | present | pass | path | failed |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for row in summary["artifacts"]:
        failed = "" if row["failed"] in (None, []) else json.dumps(row["failed"], sort_keys=True)
        lines.append(f"| {row['name']} | {row['present']} | {row['pass']} | `{row['relpath']}` | `{failed}` |")
    if summary["missing_required"]:
        lines.extend(["", "Missing required: " + ", ".join(summary["missing_required"])])
    if summary["unreadable_required"]:
        lines.extend(["", "Unreadable required: " + ", ".join(summary["unreadable_required"])])
    if summary["failed_gates"]:
        lines.extend(["", "Failed gates: " + ", ".join(summary["failed_gates"])])
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize q-proj replay artifacts and gate status.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=["auto", "preflight", "confirm"], default="auto")
    args = parser.parse_args(argv)

    summary = build_manifest(args.root, mode=args.mode)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (args.out / "report.md").write_text(render_markdown(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
