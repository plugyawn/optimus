from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from .gpu_suite import add_config_args, config_from_args, execute_specs, gpu_suite_specs


def ensure_countdown_data(path: Path, *, count: int, seed: int) -> None:
    if path.exists():
        return
    subprocess.run(
        [
            "optimus",
            "make-countdown-data",
            "--out",
            str(path),
            "--count",
            str(count),
            "--seed",
            str(seed),
        ],
        check=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute the Optimus P1024/P4096 GPU run suite.")
    add_config_args(parser, include_out=False)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--no-ensure-data", dest="ensure_data", action="store_false")
    parser.set_defaults(ensure_data=True)
    parser.add_argument("--data-count", type=int, default=1200)
    parser.add_argument("--data-seed", type=int, default=20260507)
    parser.add_argument("--execution-log", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    if args.ensure_data and not args.dry_run:
        ensure_countdown_data(config.data, count=args.data_count, seed=args.data_seed)
    rows = []

    def write_log(updated_rows: list[dict]) -> None:
        nonlocal rows
        rows = list(updated_rows)
        if args.execution_log:
            args.execution_log.parent.mkdir(parents=True, exist_ok=True)
            args.execution_log.write_text(json.dumps({"dry_run": args.dry_run, "runs": rows}, indent=2, sort_keys=True) + "\n")

    try:
        rows = execute_specs(
            gpu_suite_specs(config),
            dry_run=args.dry_run,
            skip_existing=not args.no_skip_existing,
            on_update=write_log,
        )
    except subprocess.CalledProcessError as exc:
        payload = {"dry_run": args.dry_run, "runs": rows}
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if args.execution_log:
            args.execution_log.parent.mkdir(parents=True, exist_ok=True)
            args.execution_log.write_text(text)
        print(text, end="")
        return exc.returncode
    payload = {"dry_run": args.dry_run, "runs": rows}
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.execution_log:
        args.execution_log.parent.mkdir(parents=True, exist_ok=True)
        args.execution_log.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
