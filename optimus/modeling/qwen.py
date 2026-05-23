from __future__ import annotations

SUPPORTED_QWEN_LORA_TARGETS = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}


def normalize_qwen_text_config(config):
    model_type = str(getattr(config, "model_type", ""))
    if model_type == "qwen3_vl":
        return config.text_config
    return config


def validate_qwen_lora_config(config, *, model_name: str = "model") -> None:
    model_type = str(getattr(config, "model_type", ""))
    if not (model_type.startswith("qwen2") or model_type == "qwen3_vl_text"):
        raise ValueError(
            f"{model_name} has model_type={model_type!r}; direct LoRA materialization is validated for Qwen2/Qwen3-VL text."
        )


def load_qwen_lora_config(model_name: str, *, local_files_only: bool = False):
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    config = normalize_qwen_text_config(config)
    validate_qwen_lora_config(config, model_name=model_name)
    return config


def qwen_module_prefix(config) -> str:
    model_type = str(getattr(config, "model_type", ""))
    if model_type == "qwen3_vl_text":
        return "model.language_model.layers"
    return "model.layers"


def qwen_lora_shapes(config, targets: list[str]) -> list[tuple[str, int, int]]:
    unknown = sorted(set(targets) - SUPPORTED_QWEN_LORA_TARGETS)
    if unknown:
        raise ValueError(
            "Direct Qwen adapter generation supports targets "
            f"{sorted(SUPPORTED_QWEN_LORA_TARGETS)}, got unsupported targets {unknown}."
        )
    hidden = int(config.hidden_size)
    intermediate = int(config.intermediate_size)
    layers = int(config.num_hidden_layers)
    heads = int(config.num_attention_heads)
    kv_heads = int(getattr(config, "num_key_value_heads", heads))
    head_dim = int(getattr(config, "head_dim", hidden // heads))
    kv_out = kv_heads * head_dim

    dims = {
        "q_proj": ("self_attn", hidden, hidden),
        "k_proj": ("self_attn", hidden, kv_out),
        "v_proj": ("self_attn", hidden, kv_out),
        "o_proj": ("self_attn", hidden, hidden),
        "gate_proj": ("mlp", hidden, intermediate),
        "up_proj": ("mlp", hidden, intermediate),
        "down_proj": ("mlp", intermediate, hidden),
    }
    prefix = qwen_module_prefix(config)
    shapes = []
    for layer_idx in range(layers):
        for target in targets:
            block, in_features, out_features = dims[target]
            shapes.append((f"{prefix}.{layer_idx}.{block}.{target}", in_features, out_features))
    return shapes
