from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from optimus.core.perturbations import (
    PerturbationSpec,
    parse_perturbation_key,
    perturbation_panel,
    read_perturbation_file,
    require_materialization_contract,
    write_perturbation_file,
)
from optimus.search.zeroth_order import ZerothOrderStudy, select_top_k


def test_perturbation_key_roundtrips_new_and_legacy_forms():
    spec = PerturbationSpec("isotropic", 123, 0.0075, -1, method="lora", rank=8, targets="q_proj,v_proj")

    assert parse_perturbation_key(spec.key) == spec
    assert parse_perturbation_key(spec.legacy_key) == PerturbationSpec("isotropic", 123, 0.0075, -1, method="lora")
    assert parse_perturbation_key("dense_gaussian:seed7:s0.01:sign1").method == "dense"


def test_perturbation_records_preserve_method_rank_and_targets(tmp_path: Path):
    path = tmp_path / "panel.jsonl"
    specs = perturbation_panel(
        "lora",
        "isotropic",
        4,
        0.01,
        99,
        True,
        rank=8,
        targets=("q_proj", "v_proj"),
    )

    write_perturbation_file(path, specs)
    restored = read_perturbation_file(path)

    assert len(restored) == 4
    assert {spec.method for spec in restored} == {"lora"}
    assert {spec.rank for spec in restored} == {8}
    assert restored[0].targets == ("q_proj", "v_proj")


def test_invalid_perturbation_specs_fail_fast():
    with pytest.raises(ValueError, match="method"):
        PerturbationSpec("x", 1, 0.1, method="adapter")
    with pytest.raises(ValueError, match="subspace"):
        PerturbationSpec("isotropic", 1, 0.1, method="subspace")
    with pytest.raises(ValueError, match="sign"):
        PerturbationSpec("x", 1, 0.1, sign=0)
    with pytest.raises(ValueError, match="dense perturbations"):
        PerturbationSpec("isotropic", 1, 0.1, method="dense")
    with pytest.raises(ValueError, match="dense_gaussian"):
        PerturbationSpec("dense_gaussian", 1, 0.1, method="lora")


def test_materialization_contract_requires_explicit_lora_shape():
    specs = [PerturbationSpec("isotropic", 1, 0.1, method="lora")]

    with pytest.raises(ValueError, match="missing rank"):
        require_materialization_contract(
            specs,
            backend="test",
            method="lora",
            rank=8,
            targets="q_proj,v_proj",
            require_explicit=True,
        )


def test_antithetic_panel_returns_requested_odd_population():
    specs = perturbation_panel("lora", "isotropic", 5, 0.01, 123, True, rank=8, targets="q_proj,v_proj")

    assert len(specs) == 5
    assert specs[0].sign == 1
    assert specs[1].sign == -1
    assert all(spec.rank == 8 for spec in specs)


def test_subspace_panel_roundtrips_as_subspace_method():
    specs = perturbation_panel(
        "subspace",
        "subspace_gaussian_rank_r",
        3,
        0.01,
        123,
        False,
        rank=8,
        targets="q_proj,v_proj",
    )

    assert parse_perturbation_key(specs[0].key) == specs[0]
    assert specs[0].method == "subspace"
    assert specs[0].family == "subspace_gaussian_rank_r"
    require_materialization_contract(
        specs,
        backend="test",
        method="subspace",
        rank=8,
        targets="q_proj,v_proj",
        require_explicit=True,
    )


def test_legacy_activation_subspace_names_are_not_public_api():
    with pytest.raises(ValueError, match="method"):
        PerturbationSpec(
            "activation_subspace_gaussian_rank_r",
            1,
            0.1,
            method="activation_subspace",
            rank=8,
            targets="q_proj",
        )


def test_subspace_family_still_supports_lora_export_contract():
    specs = perturbation_panel(
        "lora",
        "subspace_gaussian_rank_r",
        3,
        0.01,
        123,
        False,
        rank=8,
        targets="q_proj,v_proj",
    )

    assert specs[0].method == "lora"
    require_materialization_contract(
        specs,
        backend="adapter export",
        method="lora",
        rank=8,
        targets="q_proj,v_proj",
        require_explicit=True,
    )


def test_zeroth_order_study_is_backend_neutral():
    study = ZerothOrderStudy(method="dense", family="dense_gaussian", population=2, sigma=0.01, seed=1)
    specs = study.ask()
    study.tell(specs[0], 0.2)
    study.tell(specs[1], 0.1)

    assert all(spec.method == "dense" for spec in specs)
    assert study.result().best.candidate == specs[0].key
    assert select_top_k([{"score": 1}, {"score": 3}], score_column="score", k=1) == [{"score": 3}]


def test_perturbation_panel_cli_writes_jsonl(tmp_path: Path):
    out = tmp_path / "panel.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "optimus.cli",
            "perturbation-panel",
            "--out",
            str(out),
            "--method",
            "dense",
            "--family",
            "dense_gaussian",
            "--population",
            "2",
            "--sigma",
            "0.01",
            "--seed",
            "7",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(result.stdout)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert summary["method"] == "dense"
    assert len(rows) == 2
    assert rows[0]["key"].startswith("dense:")


def test_perturbation_panel_cli_fails_closed_for_subspace(tmp_path: Path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "optimus.cli",
            "perturbation-panel",
            "--out",
            str(tmp_path / "panel.jsonl"),
            "--method",
            "subspace",
            "--family",
            "subspace_gaussian_rank_r",
            "--population",
            "2",
            "--basis-rank",
            "128",
            "--rho-grid",
            "0.01",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "--basis-rank with --rho-grid/--sigma-w-grid" in result.stderr
