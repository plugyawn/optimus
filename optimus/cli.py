from __future__ import annotations

import argparse
import runpy
import sys
from collections.abc import Sequence


SUPPORTED_COMMANDS: dict[str, str] = {
    "backend-parity-gate": "optimus.commands.backend_parity_gate",
    "lighteval": "optimus.commands.lighteval",
    "make-countdown-data": "optimus.commands.make_countdown_data",
    "peft-search": "optimus.commands.peft_search",
    "perturbation-panel": "optimus.commands.perturbation_panel",
    "release-check": "optimus.evaluation.release",
    "run-plan": "optimus.runs.gpu_suite",
    "run-suite": "optimus.runs.gpu_suite_runner",
    "systems-report": "optimus.commands.systems_report",
    "validate-run": "optimus.evaluation.validation",
    "vllm-bench": "optimus.commands.vllm_bench",
    "vllm-halving": "optimus.commands.vllm_halving",
    "vllm-search": "optimus.commands.vllm_search",
}


COMMANDS: dict[str, str] = dict(SUPPORTED_COMMANDS)


def resolve_command(command: str) -> str:
    try:
        return COMMANDS[command]
    except KeyError as exc:
        choices = ", ".join(sorted(SUPPORTED_COMMANDS))
        raise ValueError(f"unknown Optimus command {command!r}; choose one of: {choices}") from exc


def build_parser() -> argparse.ArgumentParser:
    choices = ", ".join(sorted(SUPPORTED_COMMANDS))
    parser = argparse.ArgumentParser(
        prog="optimus",
        description="Run Optimus perturbation search, GPU validation, and reporting commands.",
        epilog="Supported commands: " + choices,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", metavar="command")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    ns = parser.parse_args(argv)
    try:
        module = resolve_command(ns.command)
    except ValueError as exc:
        parser.error(str(exc))
    sys.argv = [f"optimus {ns.command}", *ns.args]
    runpy.run_module(module, run_name="__main__", alter_sys=False)


if __name__ == "__main__":
    main()
