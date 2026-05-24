from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from optimus.evaluation.validation import check_run, gpu_suite_contracts, summary_payload
from optimus.runs.gpu_suite import GpuSuiteConfig, parse_int_tuple


FORBIDDEN_PACKAGE = "randopt_" + "lora_lab"
FORBIDDEN_REPO = "randopt-" + "lora-lab"


@dataclass(frozen=True)
class ReleaseCheck:
    name: str
    passed: bool
    detail: str


def _section(text: str, name: str) -> str:
    pattern = rf"(?ms)^\[{re.escape(name)}\]\s*(.*?)(?=^\[|\Z)"
    match = re.search(pattern, text)
    return match.group(1) if match else ""


def _quoted_value(section: str, key: str) -> str:
    match = re.search(rf'(?m)^\s*{re.escape(key)}\s*=\s*"([^"]+)"', section)
    return match.group(1) if match else ""


def _list_values(section: str, key: str) -> list[str]:
    match = re.search(rf"(?ms)^\s*{re.escape(key)}\s*=\s*\[(.*?)\]", section)
    if not match:
        return []
    return re.findall(r'"([^"]+)"', match.group(1))


def as_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def pyproject_checks(root: Path) -> list[ReleaseCheck]:
    path = root / "pyproject.toml"
    if not path.exists():
        return [ReleaseCheck("pyproject_present", False, "missing pyproject.toml")]
    text = path.read_text()
    project = _section(text, "project")
    scripts = _section(text, "project.scripts")
    packages = _section(text, "tool.setuptools.packages.find")
    package_includes = _list_values(packages, "include")
    return [
        ReleaseCheck(
            "project_name_is_optimus",
            _quoted_value(project, "name") == "optimus",
            f"project.name={_quoted_value(project, 'name')!r}",
        ),
        ReleaseCheck(
            "cli_script_is_optimus",
            _quoted_value(scripts, "optimus") == "optimus.cli:main",
            f"project.scripts.optimus={_quoted_value(scripts, 'optimus')!r}",
        ),
        ReleaseCheck(
            "package_includes_optimus",
            "optimus*" in package_includes,
            f"include={package_includes!r}",
        ),
        ReleaseCheck(
            "published_package_excludes_old_namespace",
            f"{FORBIDDEN_PACKAGE}*" not in package_includes,
            f"include={package_includes!r}",
        ),
    ]


def public_doc_checks(root: Path) -> list[ReleaseCheck]:
    docs = [
        root / "README.md",
        root / "docs" / "api.md",
        root / "docs" / "gpu_suite.md",
        root / "docs" / "index.md",
        root / "docs" / "optimus_design.md",
        root / "docs" / "release_checklist.md",
    ]
    missing = [str(path.relative_to(root)) for path in docs if not path.exists()]
    leaked: list[str] = []
    bad_patterns = [
        rf"python\s+-m\s+{FORBIDDEN_PACKAGE}",
        rf"\bfrom\s+{FORBIDDEN_PACKAGE}\b",
        rf"\bimport\s+{FORBIDDEN_PACKAGE}\b",
        rf"github\.com/[^ \n]*/{FORBIDDEN_REPO}",
    ]
    for path in docs:
        if not path.exists():
            continue
        text = path.read_text()
        for pattern in bad_patterns:
            if re.search(pattern, text):
                leaked.append(f"{path.relative_to(root)}:{pattern}")
    return [
        ReleaseCheck(
            "public_docs_present",
            not missing,
            "all required docs present" if not missing else f"missing={missing!r}",
        ),
        ReleaseCheck(
            "public_docs_do_not_promote_old_namespace",
            not leaked,
            "no old command/import examples" if not leaked else f"matches={leaked!r}",
        ),
    ]


def package_code_checks(root: Path) -> list[ReleaseCheck]:
    package_root = root / "optimus"
    if not package_root.exists():
        return [ReleaseCheck("optimus_package_source_present", False, "missing optimus package directory")]
    leaked = []
    for path in sorted(package_root.rglob("*.py")):
        text = path.read_text()
        if FORBIDDEN_PACKAGE in text or FORBIDDEN_REPO in text:
            leaked.append(str(path.relative_to(root)))
    return [
        ReleaseCheck("optimus_package_source_present", True, str(package_root)),
        ReleaseCheck(
            "optimus_package_does_not_reference_old_namespace",
            not leaked,
            "no old namespace references" if not leaked else f"files={leaked!r}",
        ),
    ]


def repo_structure_checks(root: Path) -> list[ReleaseCheck]:
    old_namespace_py = sorted(str(path.relative_to(root)) for path in (root / FORBIDDEN_PACKAGE).glob("*.py"))
    docs_archive = root / "docs" / "archive"
    scripts_archive = root / "scripts" / "archive"
    tracked_results: list[str] = []
    tracked_report_bundles: list[str] = []
    tracked_generated_data: list[str] = []
    if (root / ".git").exists():
        for label, pathspec in [
            ("results", "results"),
            ("report_bundles", "docs/reports"),
            ("generated_data", "data"),
        ]:
            result = subprocess.run(
                ["git", "ls-files", pathspec],
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                continue
            lines = [line for line in result.stdout.splitlines() if line.strip()]
            if label == "results":
                tracked_results = lines
            elif label == "report_bundles":
                tracked_report_bundles = lines
            elif label == "generated_data":
                tracked_generated_data = lines
    return [
        ReleaseCheck(
            "repo_has_no_top_level_old_namespace",
            not old_namespace_py,
            "no top-level old-namespace Python files" if not old_namespace_py else f"files={old_namespace_py[:8]!r}",
        ),
        ReleaseCheck(
            "repo_has_no_tracked_results",
            not tracked_results,
            "no tracked raw result files" if not tracked_results else f"files={tracked_results[:8]!r}",
        ),
        ReleaseCheck(
            "repo_has_no_tracked_report_bundles",
            not tracked_report_bundles,
            "no tracked docs/reports bundles" if not tracked_report_bundles else f"files={tracked_report_bundles[:8]!r}",
        ),
        ReleaseCheck(
            "repo_has_no_tracked_generated_data",
            not tracked_generated_data,
            "no tracked generated data files" if not tracked_generated_data else f"files={tracked_generated_data[:8]!r}",
        ),
        ReleaseCheck(
            "repo_has_no_archive_experiment_tree",
            not docs_archive.exists() and not scripts_archive.exists(),
            "no public archive experiment tree" if not docs_archive.exists() and not scripts_archive.exists() else "archive experiment tree present",
        ),
    ]


def git_state_checks(root: Path) -> list[ReleaseCheck]:
    if not (root / ".git").exists():
        return [ReleaseCheck("git_state_checked", True, "not a git checkout")]
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    dirty_lines = [line for line in status.stdout.splitlines() if line.strip()] if status.returncode == 0 else ["git status failed"]
    branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    upstream_detail = branch.stdout.strip() if branch.returncode == 0 else "no upstream"
    pushed = False
    if branch.returncode == 0:
        counts = subprocess.run(
            ["git", "rev-list", "--left-right", "--count", "HEAD...@{u}"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        if counts.returncode == 0:
            ahead_behind = counts.stdout.split()
            pushed = len(ahead_behind) == 2 and ahead_behind == ["0", "0"]
            upstream_detail = f"{upstream_detail} ahead={ahead_behind[0]} behind={ahead_behind[1]}"
    return [
        ReleaseCheck(
            "git_worktree_clean",
            status.returncode == 0 and not dirty_lines,
            "clean" if status.returncode == 0 and not dirty_lines else f"dirty={dirty_lines[:12]!r}",
        ),
        ReleaseCheck(
            "git_head_pushed_to_upstream",
            pushed,
            upstream_detail,
        ),
    ]


def remote_url(root: Path) -> str:
    result = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def remote_check(root: Path, url: str | None) -> ReleaseCheck:
    actual = url if url is not None else remote_url(root)
    lowered = actual.lower()
    last_path = lowered.rstrip("/").removesuffix(".git").split("/")[-1]
    passed = bool(actual) and last_path == "optimus" and FORBIDDEN_REPO not in lowered
    detail = f"origin={actual!r}" if actual else "origin remote not found"
    return ReleaseCheck("github_remote_is_optimus", passed, detail)


def systems_report_checks(systems_out: Path | None) -> list[ReleaseCheck]:
    if systems_out is None:
        return [ReleaseCheck("systems_report_checked", False, "pass --systems-out to validate report semantics")]
    report = systems_out / "report.md"
    quality = systems_out / "quality_scaling.csv"
    checks = [
        ReleaseCheck("systems_report_present", report.exists(), str(report)),
        ReleaseCheck("quality_scaling_csv_present", quality.exists(), str(quality)),
    ]
    if not quality.exists():
        checks.append(ReleaseCheck("quality_columns_are_explicit", False, "missing quality_scaling.csv"))
        return checks
    with quality.open(newline="") as f:
        reader = csv.DictReader(f)
        columns = set(reader.fieldnames or [])
    required = {
        "screen_selected_holdout_exact",
        "screen_selected_holdout_delta_vs_base",
        "promoted_holdout_oracle_exact",
        "promoted_holdout_oracle_delta_vs_base",
    }
    missing = sorted(required - columns)
    checks.append(
        ReleaseCheck(
            "quality_columns_are_explicit",
            not missing,
            "selector/oracle columns present" if not missing else f"missing={missing!r}",
        )
    )
    if report.exists():
        text = report.read_text()
        checks.append(
            ReleaseCheck(
                "report_names_selector_regret",
                "Screen-selected heldout transfer" in text or "screen-selected heldout" in text,
                str(report),
            )
        )
    parity = systems_out / "parity.csv"
    if parity.exists():
        with parity.open(newline="") as f:
            rows = list(csv.DictReader(f))
        def row_passes(row: dict) -> bool:
            required_true = ["pass", "pass_protocol", "pass_base_rows", "pass_adapter_tensors", "pass_output_diff"]
            return (
                all(str(row.get(key, "")).lower() == "true" for key in required_true)
                and as_int(row.get("n_common"), 0) > 0
                and str(row.get("trusted_name", "")).strip()
                and str(row.get("candidate_name", "")).strip()
            )

        passing = [row for row in rows if row_passes(row)]
        checks.append(
            ReleaseCheck(
                "parity_report_has_passing_gate",
                bool(passing),
                f"strict_passing={len(passing)} total={len(rows)}",
            )
        )
    else:
        checks.append(ReleaseCheck("parity_report_has_passing_gate", False, f"missing {parity}"))
    return checks


def gpu_artifact_checks(
    gpu_root: Path | None,
    systems_out: Path | None,
    populations: tuple[int, ...],
    bench_adapters: tuple[int, ...],
    run_halving: bool,
) -> list[ReleaseCheck]:
    if gpu_root is None or systems_out is None:
        return [ReleaseCheck("gpu_suite_artifacts_checked", False, "pass --gpu-root and --systems-out")]
    config = GpuSuiteConfig(
        output_root=gpu_root,
        systems_output_root=systems_out,
        populations=populations,
        bench_adapters=bench_adapters,
        run_halving=run_halving,
    )
    payload = summary_payload([check_run(contract) for contract in gpu_suite_contracts(config)])
    missing = [
        f"{check['name']}:{','.join(check['missing'])}"
        for check in payload["checks"]
        if check["missing"]
    ]
    invalid = [
        f"{check['name']}:{','.join(check['invalid'])}"
        for check in payload["checks"]
        if check.get("invalid")
    ]
    return [
        ReleaseCheck(
            "gpu_suite_artifacts_complete",
            bool(payload["pass"]),
            "all required artifacts present and valid" if payload["pass"] else f"missing={missing!r} invalid={invalid!r}",
        )
    ]


def ledger_check(root: Path) -> ReleaseCheck:
    ledger = root / ".opencode" / "prime-gpu-ledger.md"
    if not ledger.exists():
        return ReleaseCheck("prime_ledger_reports_no_active_pods", False, "missing .opencode/prime-gpu-ledger.md")
    text = ledger.read_text()
    passed = "No active Prime pods" in text and "Compute Pods (Total: 0)" in text
    return ReleaseCheck(
        "prime_ledger_reports_no_active_pods",
        passed,
        str(ledger) if passed else "ledger does not record zero active pods",
    )


def build_release_checks(
    *,
    root: Path,
    systems_out: Path | None,
    gpu_root: Path | None,
    populations: tuple[int, ...],
    bench_adapters: tuple[int, ...],
    run_halving: bool,
    remote: str | None,
) -> list[ReleaseCheck]:
    checks: list[ReleaseCheck] = []
    checks.extend(pyproject_checks(root))
    checks.extend(public_doc_checks(root))
    checks.extend(package_code_checks(root))
    checks.extend(repo_structure_checks(root))
    checks.extend(git_state_checks(root))
    checks.append(remote_check(root, remote))
    checks.extend(systems_report_checks(systems_out))
    checks.extend(gpu_artifact_checks(gpu_root, systems_out, populations, bench_adapters, run_halving))
    checks.append(ledger_check(root))
    return checks


def summary(checks: list[ReleaseCheck]) -> dict:
    return {
        "pass": all(check.passed for check in checks),
        "checks": [asdict(check) for check in checks],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check whether the Optimus repository is ready to publish.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--systems-out", type=Path)
    parser.add_argument("--gpu-root", type=Path)
    parser.add_argument("--populations", default="1024,4096")
    parser.add_argument("--bench-adapters", default="8,16,32")
    parser.add_argument("--skip-halving", action="store_true")
    parser.add_argument("--remote-url")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checks = build_release_checks(
        root=args.root,
        systems_out=args.systems_out,
        gpu_root=args.gpu_root,
        populations=parse_int_tuple(args.populations),
        bench_adapters=parse_int_tuple(args.bench_adapters),
        run_halving=not args.skip_halving,
        remote=args.remote_url,
    )
    payload = summary(checks)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    else:
        print(text, end="")
    return 1 if args.strict and not payload["pass"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
