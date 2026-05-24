"""Evaluation and report-generation entry points."""

__all__ = [
    "RunCheck",
    "RunContract",
    "backend_parity_main",
    "check_run",
    "gpu_suite_contracts",
    "lighteval_command",
    "release_summary",
]


def __getattr__(name: str):
    if name == "backend_parity_main":
        from .backend_parity import main

        return main
    if name == "lighteval_command":
        from .lighteval import build_lighteval_command

        return build_lighteval_command
    if name in {"RunCheck", "RunContract", "check_run", "gpu_suite_contracts"}:
        from . import validation

        return getattr(validation, name)
    if name == "release_summary":
        from .release import summary

        return summary
    raise AttributeError(name)
