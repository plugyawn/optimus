import json
from pathlib import Path

from randopt_lora_lab.replay_manifest import build_manifest


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_preflight_manifest_requires_source_panel_shortlist_and_preflight(tmp_path: Path):
    write_json(tmp_path / "dense" / "summary.json", {"kind": "dense_search"})
    write_json(tmp_path / "vllm" / "summary.json", {"kind": "vllm_search"})
    write_jsonl(tmp_path / "shortlist_top4.jsonl", [{"candidate": "activation_spectral_lora_c2:seed0"}])
    write_json(tmp_path / "preflight_summary.json", {"kind": "existing_vllm_shortlist_confirmation_preflight", "pass": True})

    summary = build_manifest(tmp_path, mode="preflight")

    assert summary["artifact_complete"] is True
    assert summary["method_pass"] is False
    assert summary["missing_required"] == []
    assert summary["unreadable_required"] == []
    assert summary["failed_gates"] == []
    shortlist = next(row for row in summary["artifacts"] if row["name"] == "shortlist")
    assert shortlist["present"] is True
    assert shortlist["rows"] == 1
    assert shortlist["relpath"] == "shortlist_top4.jsonl"


def test_confirm_manifest_separates_missing_artifacts_from_failed_gates(tmp_path: Path):
    write_json(tmp_path / "dense" / "summary.json", {"kind": "dense_search"})
    write_json(tmp_path / "vllm" / "summary.json", {"kind": "vllm_search"})
    write_jsonl(tmp_path / "shortlist_top4.jsonl", [{"candidate": "activation_spectral_lora_c2:seed0"}])
    write_json(tmp_path / "preflight_summary.json", {"pass": True})
    write_json(tmp_path / "confirmed" / "summary.json", {"kind": "search"})
    write_json(tmp_path / "confirmed" / "validity" / "summary.json", {"pass": True})
    write_json(tmp_path / "shortlist_dense_confirmation" / "summary.json", {"gate": {"pass": False, "failed": ["dense_best_recovered"]}})
    write_json(tmp_path / "family_state_provenance_audit" / "summary.json", {"pass": True})
    write_json(tmp_path / "search_quality_confirmation" / "summary.json", {"gate": {"pass": True}})

    summary = build_manifest(tmp_path, mode="confirm")

    assert summary["artifact_complete"] is False
    assert summary["missing_required"] == ["current_goal_audit"]
    assert summary["unreadable_required"] == []
    assert summary["failed_gates"] == ["shortlist_dense_confirmation"]
    assert summary["method_pass"] is False


def test_manifest_marks_corrupt_required_json_unreadable(tmp_path: Path):
    (tmp_path / "dense").mkdir()
    (tmp_path / "dense" / "summary.json").write_text("{not json")
    write_json(tmp_path / "vllm" / "summary.json", {"kind": "vllm_search"})
    write_jsonl(tmp_path / "shortlist_top4.jsonl", [{"candidate": "activation_spectral_lora_c2:seed0"}])
    write_json(tmp_path / "preflight_summary.json", {"pass": True})

    summary = build_manifest(tmp_path, mode="preflight")

    assert summary["artifact_complete"] is False
    assert summary["missing_required"] == []
    assert summary["unreadable_required"] == ["dense_summary"]
    dense = next(row for row in summary["artifacts"] if row["name"] == "dense_summary")
    assert dense["present"] is True
    assert dense["readable"] is False
    assert "JSONDecodeError" in dense["error"]


def test_confirm_manifest_method_pass_requires_current_goal_audit_pass(tmp_path: Path):
    write_json(tmp_path / "dense" / "summary.json", {"kind": "dense_search"})
    write_json(tmp_path / "vllm" / "summary.json", {"kind": "vllm_search"})
    write_jsonl(tmp_path / "shortlist_top4.jsonl", [{"candidate": "activation_spectral_lora_c2:seed0"}])
    write_json(tmp_path / "preflight_summary.json", {"pass": True})
    write_json(tmp_path / "confirmed" / "summary.json", {"kind": "search"})
    write_json(tmp_path / "confirmed" / "validity" / "summary.json", {"pass": True})
    write_json(tmp_path / "shortlist_dense_confirmation" / "summary.json", {"gate": {"pass": True}})
    write_json(tmp_path / "family_state_provenance_audit" / "summary.json", {"pass": True})
    write_json(tmp_path / "search_quality_confirmation" / "summary.json", {"gate": {"pass": True}})
    write_json(tmp_path / "current_goal_audit" / "summary.json", {"pass": True})

    summary = build_manifest(tmp_path, mode="confirm")

    assert summary["artifact_complete"] is True
    assert summary["unreadable_required"] == []
    assert summary["failed_gates"] == []
    assert summary["method_pass"] is True
