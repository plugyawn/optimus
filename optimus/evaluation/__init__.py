"""Evaluation and report-generation entry points."""

__all__ = [
    "RunCheck",
    "RunContract",
    "backend_parity_main",
    "build_systems_report",
    "check_run",
    "compare_backends",
    "gpu_suite_contracts",
    "release_summary",
]


def __getattr__(name: str):
    if name == "build_systems_report":
        from .reports import build_systems_report

        return build_systems_report
    if name == "backend_parity_main":
        from .backend_parity import main

        return main
    if name == "compare_backends":
        from .compare import compare

        return compare
    if name in {"RunCheck", "RunContract", "check_run", "gpu_suite_contracts"}:
        from . import validation

        return getattr(validation, name)
    if name == "release_summary":
        from .release import summary

        return summary
    raise AttributeError(name)
