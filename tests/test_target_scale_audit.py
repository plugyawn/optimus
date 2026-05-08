import math

from randopt_lora_lab.target_scale_audit import (
    analyze,
    flat_spectral_lora_dense_ratio,
    qwen2_attention_shapes,
    scale_for_dense_ratio,
)


def test_qwen2_rank32_q_c2_has_half_dense_frobenius_ratio():
    shapes = {shape.name: shape for shape in qwen2_attention_shapes()}

    q_ratio = flat_spectral_lora_dense_ratio(scale=2.0, rank=32, shape=shapes["q_proj"])
    v_ratio = flat_spectral_lora_dense_ratio(scale=2.0, rank=32, shape=shapes["v_proj"])

    assert math.isclose(q_ratio, 0.5, rel_tol=1e-6)
    assert math.isclose(v_ratio, 0.95710678, rel_tol=1e-5)


def test_scale_for_dense_ratio_matches_reference_ratio():
    shapes = {shape.name: shape for shape in qwen2_attention_shapes()}
    reference = flat_spectral_lora_dense_ratio(scale=2.0, rank=32, shape=shapes["q_proj"])
    v_scale = scale_for_dense_ratio(target_ratio=reference, rank=32, shape=shapes["v_proj"])

    assert math.isclose(v_scale, 1.044815, rel_tol=1e-4)
    assert math.isclose(
        flat_spectral_lora_dense_ratio(scale=v_scale, rank=32, shape=shapes["v_proj"]),
        reference,
        rel_tol=1e-6,
    )


def test_analyze_emits_target_scaled_family():
    summary = analyze(
        qwen2_attention_shapes(),
        rank=32,
        reference_target="q_proj",
        reference_scale=2.0,
    )

    assert math.isclose(summary["reference_lora_over_dense"], 0.5, rel_tol=1e-12)
    assert summary["matched_reference_family"].startswith("activation_spectral_lora_tscale_q2_")
    assert "_v1p045_" in summary["matched_reference_family"]
