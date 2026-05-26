from __future__ import annotations

import argparse

import pytest

from scripts.bench_subspace_delta_kernels import DEFAULT_SHAPES, Shape, parse_shape


def test_parse_shape_accepts_comma_and_x_forms():
    assert parse_shape("64,128,4096,16") == Shape(rows=64, rank=128, output_dim=4096, candidates=16)
    assert parse_shape("64x128x4096x16") == Shape(rows=64, rank=128, output_dim=4096, candidates=16)


def test_parse_shape_rejects_bad_shapes():
    with pytest.raises(argparse.ArgumentTypeError):
        parse_shape("64,128,4096")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_shape("64,128,4096,0")


def test_default_shapes_cover_small_and_large_output_regimes():
    assert min(shape.output_dim for shape in DEFAULT_SHAPES) <= 1024
    assert max(shape.output_dim for shape in DEFAULT_SHAPES) >= 4096
    assert {shape.rank for shape in DEFAULT_SHAPES} >= {64, 128}
