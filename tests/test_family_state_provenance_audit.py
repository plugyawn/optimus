import json
from pathlib import Path

from randopt_lora_lab.family_state_provenance_audit import audit_run, discover_roots, preflight_existing_panel, run_audit


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def test_audit_passes_when_confirmed_copies_vllm_family_state(tmp_path: Path):
    root = tmp_path / "run"
    write_json(root / "vllm" / "summary.json", {"family": "activation_spectral_lora_c2"})
    write_json(
        root / "confirmed" / "summary.json",
        {"family": "activation_spectral_lora_c2", "family_state_file": str(root / "vllm" / "family_state.pt")},
    )
    write_bytes(root / "vllm" / "family_state.pt", b"same-state")
    write_bytes(root / "confirmed" / "family_state.pt", b"same-state")
    write_json(root / "confirmed" / "family_state_summary.json", {"kind": "loaded_family_state", "source": str(root / "vllm" / "family_state.pt")})

    summary = audit_run(root)

    assert summary["pass"] is True
    assert summary["requires_family_state"] is True
    assert summary["failed"] == []


def test_audit_fails_activation_run_without_confirmed_family_state(tmp_path: Path):
    root = tmp_path / "run"
    write_json(root / "vllm" / "summary.json", {"family": "activation_spectral_lora_c2"})
    write_json(root / "confirmed" / "summary.json", {"family": "activation_spectral_lora_c2"})
    write_bytes(root / "vllm" / "family_state.pt", b"state")

    summary = audit_run(root)

    assert summary["pass"] is False
    assert "confirmed_family_state_present" in summary["failed"]
    assert "confirmed_family_state_summary_loaded" in summary["failed"]


def test_audit_skips_non_activation_runs_without_family_state(tmp_path: Path):
    root = tmp_path / "run"
    write_json(root / "vllm" / "summary.json", {"family": "sparse_low_rank_lora_d0p125"})
    write_json(root / "confirmed" / "summary.json", {"family": "sparse_low_rank_lora_d0p125"})

    summary = audit_run(root)

    assert summary["pass"] is True
    assert summary["requires_family_state"] is False


def test_discovery_and_run_audit(tmp_path: Path):
    root = tmp_path / "results" / "run"
    write_json(root / "vllm" / "summary.json", {"family": "activation_spectral_lora_c2"})
    write_json(root / "confirmed" / "summary.json", {"family": "activation_spectral_lora_c2"})
    write_bytes(root / "vllm" / "family_state.pt", b"state")

    roots = discover_roots(tmp_path / "results")
    summary = run_audit(roots)

    assert roots == [root]
    assert summary["pass"] is False
    assert summary["failed"] == [str(root)]


def test_existing_panel_preflight_passes_for_matching_shortlist_and_state(tmp_path: Path):
    source = tmp_path / "source"
    out = tmp_path / "out"
    candidate = "activation_spectral_lora_c2:seed1:s0.002:sign1"
    write_json(source / "vllm" / "summary.json", {"family": "activation_spectral_lora_c2"})
    write_json(out / "vllm" / "summary.json", {"family": "activation_spectral_lora_c2"})
    write_json(out / "dense" / "summary.json", {"family": "dense_gaussian"})
    write_bytes(source / "vllm" / "family_state.pt", b"state")
    write_bytes(out / "vllm" / "family_state.pt", b"state")
    write_jsonl(out / "vllm" / "candidate_summary.jsonl", [{"candidate": candidate}])
    write_jsonl(out / "shortlist_top1.jsonl", [{"candidate": candidate, "selector_union_policy": "default_exact"}])

    summary = preflight_existing_panel(source, out, family="activation_spectral_lora_c2", shortlist_k=1)

    assert summary["pass"] is True
    assert summary["shortlist_policy"] == "default_exact"
    assert summary["source_family_state_sha256"] == summary["vllm_family_state_sha256"]


def test_existing_panel_preflight_fails_wrong_family_and_state_mismatch(tmp_path: Path):
    source = tmp_path / "source"
    out = tmp_path / "out"
    write_json(out / "dense" / "summary.json", {"family": "dense_gaussian"})
    write_bytes(source / "vllm" / "family_state.pt", b"source")
    write_bytes(out / "vllm" / "family_state.pt", b"out")
    write_jsonl(out / "vllm" / "candidate_summary.jsonl", [{"candidate": "other_family:seed1:s0.002:sign1"}])
    write_jsonl(out / "shortlist_top1.jsonl", [{"candidate": "other_family:seed1:s0.002:sign1"}])

    summary = preflight_existing_panel(source, out, family="activation_spectral_lora_c2", shortlist_k=1)

    assert summary["pass"] is False
    assert "shortlist_candidates_match_family" in summary["failed"]
    assert "copied_family_state_matches_source" in summary["failed"]
