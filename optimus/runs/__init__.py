"""Run specifications for Optimus workloads."""

__all__ = [
    "GpuSuiteConfig",
    "RunSpec",
    "execute_specs",
    "gpu_suite_specs",
    "plan_payload",
]


def __getattr__(name: str):
    if name in __all__:
        from . import gpu_suite

        return getattr(gpu_suite, name)
    raise AttributeError(name)
