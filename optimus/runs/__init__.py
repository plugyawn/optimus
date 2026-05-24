"""Run specifications for Optimus workloads."""

__all__ = [
    "GpuSuiteConfig",
    "ExperimentKey",
    "RunSpec",
    "RunRecord",
    "execute_specs",
    "gpu_suite_specs",
    "plan_payload",
]


def __getattr__(name: str):
    if name in {"ExperimentKey", "RunRecord"}:
        from optimus.core import ExperimentKey, RunRecord

        return {"ExperimentKey": ExperimentKey, "RunRecord": RunRecord}[name]
    if name in __all__:
        from . import gpu_suite

        return getattr(gpu_suite, name)
    raise AttributeError(name)
