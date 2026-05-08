import json
from pathlib import Path

from randopt_lora_lab.family_state_provenance_audit import audit_run, discover_roots, run_audit


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


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
