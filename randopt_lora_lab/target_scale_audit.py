from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModuleShape:
    name: str
    out_features: int
    in_features: int


def qwen2_attention_shapes(
    *,
    hidden_size: int = 2048,
    num_attention_heads: int = 16,
    num_key_value_heads: int = 2,
    head_dim: int | None = None,
) -> list[ModuleShape]:
    head_dim = int(head_dim or hidden_size // num_attention_heads)
    kv_out = int(num_key_value_heads) * head_dim
    return [
        ModuleShape("q_proj", int(hidden_size), int(hidden_size)),
        ModuleShape("k_proj", kv_out, int(hidden_size)),
        ModuleShape("v_proj", kv_out, int(hidden_size)),
        ModuleShape("o_proj", int(hidden_size), int(hidden_size)),
    ]


def effective_rank(rank: int, shape: ModuleShape) -> int:
    return max(0, min(int(rank), int(shape.out_features), int(shape.in_features)))


def flat_spectral_lora_dense_ratio(*, scale: float, rank: int, shape: ModuleShape) -> float:
    k = effective_rank(rank, shape)
    if k == 0:
        return 0.0
    numerator = float(scale) * (math.sqrt(shape.out_features) + math.sqrt(shape.in_features)) * math.sqrt(k)
    denominator = math.sqrt(shape.out_features * shape.in_features)
    return numerator / denominator


def scale_for_dense_ratio(*, target_ratio: float, rank: int, shape: ModuleShape) -> float:
    k = effective_rank(rank, shape)
    if k == 0:
        raise ValueError(f"cannot scale zero-rank shape {shape}")
    numerator = float(target_ratio) * math.sqrt(shape.out_features * shape.in_features)
    denominator = (math.sqrt(shape.out_features) + math.sqrt(shape.in_features)) * math.sqrt(k)
    return numerator / denominator


def family_float(value: float, *, digits: int = 3) -> str:
    text = f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def target_key(name: str) -> str:
    return {
        "q_proj": "q",
        "k_proj": "k",
        "v_proj": "v",
        "o_proj": "o",
        "gate_proj": "gate",
        "up_proj": "up",
        "down_proj": "down",
    }.get(name, name.replace("_proj", ""))


def analyze(
    shapes: list[ModuleShape],
    *,
    rank: int,
    reference_target: str,
    reference_scale: float,
) -> dict:
    by_name = {shape.name: shape for shape in shapes}
    if reference_target not in by_name:
        raise ValueError(f"reference target {reference_target!r} not in shapes: {sorted(by_name)}")
    reference_shape = by_name[reference_target]
    reference_ratio = flat_spectral_lora_dense_ratio(scale=reference_scale, rank=rank, shape=reference_shape)
    rows = []
    reference_update_norm_unit_sigma = (
        reference_scale
        * (math.sqrt(reference_shape.out_features) + math.sqrt(reference_shape.in_features))
        * math.sqrt(effective_rank(rank, reference_shape))
    )
    for shape in shapes:
        same_scale_ratio = flat_spectral_lora_dense_ratio(scale=reference_scale, rank=rank, shape=shape)
        scale_match_reference = scale_for_dense_ratio(target_ratio=reference_ratio, rank=rank, shape=shape)
        scale_match_dense = scale_for_dense_ratio(target_ratio=1.0, rank=rank, shape=shape)
        matched_update_norm_unit_sigma = (
            scale_match_reference
            * (math.sqrt(shape.out_features) + math.sqrt(shape.in_features))
            * math.sqrt(effective_rank(rank, shape))
        )
        rows.append(
            {
                **asdict(shape),
                "effective_rank": effective_rank(rank, shape),
                "same_scale": reference_scale,
                "same_scale_lora_over_dense": same_scale_ratio,
                "scale_for_reference_ratio": scale_match_reference,
                "scale_for_lora_over_dense_1": scale_match_dense,
                "matched_update_norm_unit_sigma": matched_update_norm_unit_sigma,
            }
        )
    matched_tokens = [
        f"{target_key(row['name'])}{family_float(row['scale_for_reference_ratio'])}"
        for row in rows
    ]
    total_matched_update_norm = math.sqrt(sum(float(row["matched_update_norm_unit_sigma"]) ** 2 for row in rows))
    global_budget_multiplier = reference_update_norm_unit_sigma / total_matched_update_norm
    global_budget_tokens = [
        f"{target_key(row['name'])}{family_float(row['scale_for_reference_ratio'] * global_budget_multiplier)}"
        for row in rows
    ]
    return {
        "kind": "target_scale_audit",
        "rank": rank,
        "reference_target": reference_target,
        "reference_scale": reference_scale,
        "reference_lora_over_dense": reference_ratio,
        "reference_update_norm_unit_sigma": reference_update_norm_unit_sigma,
        "matched_total_update_norm_unit_sigma": total_matched_update_norm,
        "matched_total_over_reference_update_norm": total_matched_update_norm / reference_update_norm_unit_sigma,
        "global_budget_multiplier": global_budget_multiplier,
        "shapes": [asdict(shape) for shape in shapes],
        "rows": rows,
        "matched_reference_family": "activation_spectral_lora_tscale_" + "_".join(matched_tokens),
        "global_budget_family": "activation_spectral_lora_tscale_" + "_".join(global_budget_tokens),
    }


def fmt(value: float) -> str:
    return f"{value:.6g}"


def render_report(summary: dict) -> str:
    lines = [
        "# Target Scale Audit",
        "",
        f"rank: `{summary['rank']}`",
        f"reference: `{summary['reference_target']}` at c=`{summary['reference_scale']}`",
        f"reference LoRA/dense Frobenius ratio: `{fmt(summary['reference_lora_over_dense'])}`",
        "",
        "| target | shape | effective rank | ratio at reference c | c for reference ratio | c for LoRA/dense=1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["rows"]:
        shape = f"{row['out_features']}x{row['in_features']}"
        lines.append(
            f"| {row['name']} | {shape} | {row['effective_rank']} | "
            f"{fmt(row['same_scale_lora_over_dense'])} | {fmt(row['scale_for_reference_ratio'])} | "
            f"{fmt(row['scale_for_lora_over_dense_1'])} |"
        )
    lines.extend(
        [
            "",
            "Matched-reference family:",
            "",
            "```text",
            summary["matched_reference_family"],
            "```",
            "",
            "Global-budget matched family:",
            "",
            "```text",
            summary["global_budget_family"],
            "```",
            "",
            f"Matched-reference total/update norm is `{fmt(summary['matched_total_over_reference_update_norm'])}`x "
            "the single-reference target. The global-budget family scales all listed targets down by "
            f"`{fmt(summary['global_budget_multiplier'])}` to keep total update Frobenius matched to the reference arm.",
            "",
            "The current flat activation-spectral rule is shape-dependent. A single c value is not a "
            "fair comparison across q/k/v/o when key/value projections have smaller output width.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_targets(text: str, shapes: list[ModuleShape]) -> list[ModuleShape]:
    by_name = {shape.name: shape for shape in shapes}
    targets = [item.strip() for item in text.split(",") if item.strip()]
    missing = [item for item in targets if item not in by_name]
    if missing:
        raise ValueError(f"unknown targets {missing}; available targets are {sorted(by_name)}")
    return [by_name[item] for item in targets]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit flat activation-spectral LoRA scaling across target shapes.")
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--num-attention-heads", type=int, default=16)
    parser.add_argument("--num-key-value-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=0)
    parser.add_argument("--targets", default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--reference-target", default="q_proj")
    parser.add_argument("--reference-scale", type=float, default=2.0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    shapes = qwen2_attention_shapes(
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        head_dim=args.head_dim or None,
    )
    selected = parse_targets(args.targets, shapes)
    summary = analyze(
        selected,
        rank=args.rank,
        reference_target=args.reference_target,
        reference_scale=args.reference_scale,
    )
    report = render_report(summary)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        (args.out / "report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
