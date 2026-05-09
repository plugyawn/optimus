import json
import os
from pathlib import Path
import subprocess
import sys


def test_qproj_c2_replay_defaults_to_preflight_mode():
    text = Path("scripts/run_qproj_c2_exact_replay.sh").read_text()

    assert "MODE=${MODE:-preflight}" in text
    assert "export PREFLIGHT_ONLY=1" in text
    assert "MODE=confirm" in text
    assert "scripts/run_existing_vllm_shortlist_confirmation.sh" in text
    assert 'if [[ "$MODE" == "confirm" && "$RUN_GOAL_AUDIT" == "1" ]]' in text
    assert 'QPROJ_REPLAY_ROOT="$OUT_ROOT" OUT="$GOAL_AUDIT_OUT" scripts/run_current_goal_audit.sh' in text
    assert "randopt_lora_lab.score_sanity_audit" in text
    assert "randopt_lora_lab.replay_manifest" in text
    assert '--mode "$MODE"' in text


def test_qproj_c2_replay_targets_saved_basis_shortlist():
    text = Path("scripts/run_qproj_c2_exact_replay.sh").read_text()

    assert "results/qproj_c2_vllm_shortlist_p64" in text
    assert "activation_spectral_lora_c2" in text
    assert "TARGETS=${TARGETS:-q_proj}" in text
    assert "CONFIRM_MAX_DENSE_REGRET=${CONFIRM_MAX_DENSE_REGRET:-0.015625}" in text


def test_qproj_c2_corrected_confirmation_is_guarded_and_uses_healthy_prompts():
    text = Path("scripts/run_qproj_c2_corrected_confirmation.sh").read_text()

    assert "MODE=${MODE:-preflight}" in text
    assert "MODE=confirm" in text
    assert "scripts/run_vllm_shortlist_confirmation.sh" in text
    assert "OUT_ROOT=${OUT_ROOT:-results/qproj_c2_vllm_shortlist_p64_default_reordered}" in text
    assert "FAMILY=${FAMILY:-activation_spectral_lora_c2}" in text
    assert "TARGETS=${TARGETS:-q_proj}" in text
    assert "SHORTLIST_POLICY=${SHORTLIST_POLICY:-default_exact}" in text
    assert "VLLM_PROMPT_VARIANTS=${VLLM_PROMPT_VARIANTS:-default,reordered}" in text
    assert "VLLM_REQUIRE_ALL_PROMPT_VARIANTS_VALID=${VLLM_REQUIRE_ALL_PROMPT_VARIANTS_VALID:-1}" in text
    assert "qproj_c2_corrected_confirmation_preflight" in text
    assert "preflight_summary.json" in text
    assert 'QPROJ_REPLAY_ROOT="$OUT_ROOT" OUT="$GOAL_AUDIT_OUT" scripts/run_current_goal_audit.sh' in text
    assert "current goal audit failed; continuing to replay manifest" in text
    assert "randopt_lora_lab.replay_manifest" in text


def test_qproj_c2_corrected_preflight_writes_summary(tmp_path):
    out_root = tmp_path / "qproj_preflight"
    env = os.environ.copy()
    env.update({
        "MODE": "preflight",
        "OUT_ROOT": str(out_root),
        "PYTHON": sys.executable,
    })

    result = subprocess.run(
        ["bash", "scripts/run_qproj_c2_corrected_confirmation.sh"],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )

    summary_path = out_root / "preflight_summary.json"
    summary = json.loads(summary_path.read_text())
    assert "preflight complete:" in result.stdout
    assert summary["kind"] == "qproj_c2_corrected_confirmation_preflight"
    assert summary["pass"] is True
    assert summary["mode"] == "preflight"
    assert summary["config"]["OUT_ROOT"] == str(out_root)
    assert summary["config"]["VLLM_PROMPT_VARIANTS"] == "default,reordered"


def test_current_goal_audit_script_is_non_gpu_and_dense_referenced():
    text = Path("scripts/run_current_goal_audit.sh").read_text()

    assert "randopt_lora_lab.goal_audit" in text
    assert "results/current_goal_audit_current" in text
    assert "QPROJ_REPLAY_ROOT=${QPROJ_REPLAY_ROOT:-results/qproj_c2_vllm_shortlist_p64_default_reordered}" in text
    assert "DENSE_CONFIRMATION_GATE=${DENSE_CONFIRMATION_GATE:-$QPROJ_REPLAY_ROOT/shortlist_dense_confirmation/summary.json}" in text
    assert "SEARCH_QUALITY_CONFIRMATION=${SEARCH_QUALITY_CONFIRMATION:-$QPROJ_REPLAY_ROOT/search_quality_confirmation/summary.json}" in text
    assert "FAMILY_STATE_PROVENANCE=${FAMILY_STATE_PROVENANCE:-$QPROJ_REPLAY_ROOT/family_state_provenance_audit/summary.json}" in text
    assert "results/family_state_provenance_audit_current" not in text
    assert "--dense-confirmation-gate \"$DENSE_CONFIRMATION_GATE\"" in text
    assert "--search-quality-confirmation \"$SEARCH_QUALITY_CONFIRMATION\"" in text
    assert "MODE=confirm" not in text
    assert "experiments search" not in text


def test_existing_replay_keeps_diagnostics_after_gate_failures():
    text = Path("scripts/run_existing_vllm_shortlist_confirmation.sh").read_text()

    assert "confirmed validity gate failed; continuing to write downstream diagnostics" in text
    assert "family-state provenance audit failed unexpectedly; continuing to search-quality diagnostics" in text
    assert "--no-fail" in text
    assert "randopt_lora_lab.shortlist_dense_confirmation" in text
    assert "randopt_lora_lab.search_quality_confirmation" in text


def test_vllm_shortlist_confirmation_keeps_diagnostics_after_gate_failures():
    text = Path("scripts/run_vllm_shortlist_confirmation.sh").read_text()

    assert "dense validity gate failed; continuing to write downstream diagnostics" in text
    assert "confirmed validity gate failed; continuing to write downstream diagnostics" in text
    assert "family-state provenance audit failed unexpectedly; continuing to score-sanity diagnostics" in text
    assert 'if ! "$PYTHON" -m randopt_lora_lab.result_validity' in text
    assert "--no-fail" in text
    assert "randopt_lora_lab.search_quality_confirmation" in text
    assert "randopt_lora_lab.score_sanity_audit" in text


def test_vllm_confirmation_wrappers_require_all_prompt_variants_valid_by_default():
    for path in [
        Path("scripts/run_vllm_shortlist_confirmation.sh"),
        Path("scripts/run_spectral_vllm_confirmation.sh"),
    ]:
        text = path.read_text()
        assert "default,reordered,xml" not in text
        assert "default,reordered" in text
        assert "VLLM_REQUIRE_ALL_PROMPT_VARIANTS_VALID=${VLLM_REQUIRE_ALL_PROMPT_VARIANTS_VALID:-1}" in text
        assert "--require-all-prompt-variants-valid" in text


def test_vllm_shortlist_confirmation_writes_quality_and_score_sanity_gates():
    text = Path("scripts/run_vllm_shortlist_confirmation.sh").read_text()

    assert "RUN_SEARCH_QUALITY=${RUN_SEARCH_QUALITY:-1}" in text
    assert "RUN_SCORE_SANITY=${RUN_SCORE_SANITY:-1}" in text
    assert "randopt_lora_lab.search_quality_confirmation" in text
    assert "randopt_lora_lab.score_sanity_audit" in text
