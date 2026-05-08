from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def activation_family(family: str | None) -> bool:
    return bool(
        family
        and (
            family.startswith("activation_spectral_lora")
            or family.startswith("activation_projected_gaussian_rank_r")
            or family.startswith("activation_generalized_projected_gaussian_rank_r")
            or family.startswith("activation_generalized_spectral_lora")
        )
    )


def discover_roots(results_root: Path) -> list[Path]:
    roots = {path.parent.parent for path in results_root.rglob("vllm/family_state.pt")}
    return sorted(roots)


def check(path: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"check": path, "passed": bool(passed), "detail": detail}


def audit_run(root: Path) -> dict[str, Any]:
    vllm = root / "vllm"
    confirmed = root / "confirmed"
    vllm_summary = read_json(vllm / "summary.json")
    confirmed_summary = read_json(confirmed / "summary.json")
    vllm_family = vllm_summary.get("family")
    confirmed_family = confirmed_summary.get("family")
    family = confirmed_family or vllm_family
    vllm_state = vllm / "family_state.pt"
    confirmed_state = confirmed / "family_state.pt"
    confirmed_state_summary_path = confirmed / "family_state_summary.json"
    confirmed_state_summary = read_json(confirmed_state_summary_path)
    requires_state = vllm_state.exists() and activation_family(str(family))
    checks = [
        check("vllm_family_state_present", vllm_state.exists(), str(vllm_state)),
        check("confirmed_summary_present", (confirmed / "summary.json").exists(), str(confirmed / "summary.json")),
    ]
    if requires_state:
        confirmed_state_hash = sha256(confirmed_state)
        vllm_state_hash = sha256(vllm_state)
        summary_source = confirmed_state_summary.get("source")
        summary_source_name = None if summary_source is None else Path(str(summary_source)).name
        checks.extend(
            [
                check(
                    "confirmed_summary_family_state_file_present",
                    bool(confirmed_summary.get("family_state_file")),
                    {"family_state_file": confirmed_summary.get("family_state_file")},
                ),
                check("confirmed_family_state_present", confirmed_state.exists(), str(confirmed_state)),
                check(
                    "confirmed_family_state_matches_vllm",
                    confirmed_state_hash is not None and confirmed_state_hash == vllm_state_hash,
                    {"confirmed_sha256": confirmed_state_hash, "vllm_sha256": vllm_state_hash},
                ),
                check(
                    "confirmed_family_state_summary_present",
                    confirmed_state_summary_path.exists(),
                    str(confirmed_state_summary_path),
                ),
                check(
                    "confirmed_family_state_summary_loaded",
                    confirmed_state_summary.get("kind") == "loaded_family_state",
                    {"kind": confirmed_state_summary.get("kind")},
                ),
                check(
                    "confirmed_family_state_summary_points_to_state",
                    summary_source_name == "family_state.pt",
                    {"source": summary_source},
                ),
            ]
        )
    failed = [row["check"] for row in checks if requires_state and not row["passed"]]
    return {
        "root": str(root),
        "family": family,
        "vllm_family": vllm_family,
        "confirmed_family": confirmed_family,
        "requires_family_state": requires_state,
        "pass": not failed,
        "failed": failed,
        "checks": checks,
    }


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Family State Provenance Audit",
        "",
        f"Gate: **{'PASS' if summary['pass'] else 'FAIL'}**",
        "",
        "| run | required | pass | failed |",
        "| --- | ---: | ---: | --- |",
    ]
    for row in summary["runs"]:
        failed = ", ".join(row["failed"]) if row["failed"] else ""
        lines.append(f"| `{row['root']}` | {row['requires_family_state']} | {row['pass']} | {failed} |")
    lines.append("")
    lines.append("## Details")
    for row in summary["runs"]:
        lines.extend(
            [
                "",
                f"### `{row['root']}`",
                "",
                f"- family: `{row['family']}`",
                f"- requires family state: `{row['requires_family_state']}`",
                f"- pass: `{row['pass']}`",
                "",
                "| check | pass | detail |",
                "| --- | ---: | --- |",
            ]
        )
        for check_row in row["checks"]:
            detail = json.dumps(check_row["detail"], sort_keys=True)
            lines.append(f"| `{check_row['check']}` | {check_row['passed']} | `{detail}` |")
    return "\n".join(lines) + "\n"


def run_audit(roots: list[Path]) -> dict[str, Any]:
    runs = [audit_run(root) for root in roots]
    failed = [row["root"] for row in runs if not row["pass"]]
    return {
        "kind": "family_state_provenance_audit",
        "pass": not failed,
        "failed": failed,
        "runs": runs,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, action="append", default=[])
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args(argv)

    roots = args.root or discover_roots(args.results_root)
    summary = run_audit(roots)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (args.out / "report.md").write_text(render_report(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["pass"] or args.no_fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
