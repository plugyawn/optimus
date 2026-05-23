"""Compatibility entrypoint for Optimus vLLM LoRA benchmarking."""

from optimus.modeling import AdapterSpec, parse_targets, save_seed_adapter
from optimus.modeling.qwen import qwen_lora_shapes
from optimus.serving.benchmark import (
    DEFAULT_MODEL,
    DEFAULT_TARGETS,
    SUPPORTED_QWEN_TARGETS,
    build_parser,
    diagnostic_payload,
    main,
    make_adapter_specs,
    reset_jsonl_outputs,
    run_benchmark,
)
from optimus.serving.runtime import (
    import_vllm_lora_request,
    make_sampling_params,
    package_version,
    score_mixed_rows,
    score_rows,
    timed_generate,
    write_json,
    write_jsonl,
)

__all__ = [
    "AdapterSpec",
    "DEFAULT_MODEL",
    "DEFAULT_TARGETS",
    "SUPPORTED_QWEN_TARGETS",
    "build_parser",
    "diagnostic_payload",
    "import_vllm_lora_request",
    "main",
    "make_adapter_specs",
    "make_sampling_params",
    "package_version",
    "parse_targets",
    "qwen_lora_shapes",
    "reset_jsonl_outputs",
    "run_benchmark",
    "save_seed_adapter",
    "score_mixed_rows",
    "score_rows",
    "timed_generate",
    "write_json",
    "write_jsonl",
]


if __name__ == "__main__":
    raise SystemExit(main())
