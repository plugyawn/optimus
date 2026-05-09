from pathlib import Path


def test_qproj_c2_replay_defaults_to_preflight_mode():
    text = Path("scripts/run_qproj_c2_exact_replay.sh").read_text()

    assert "MODE=${MODE:-preflight}" in text
    assert "export PREFLIGHT_ONLY=1" in text
    assert "MODE=confirm" in text
    assert "exec scripts/run_existing_vllm_shortlist_confirmation.sh" in text


def test_qproj_c2_replay_targets_saved_basis_shortlist():
    text = Path("scripts/run_qproj_c2_exact_replay.sh").read_text()

    assert "results/qproj_c2_vllm_shortlist_p64" in text
    assert "activation_spectral_lora_c2" in text
    assert "TARGETS=${TARGETS:-q_proj}" in text
    assert "CONFIRM_MAX_DENSE_REGRET=${CONFIRM_MAX_DENSE_REGRET:-0.015625}" in text


def test_current_goal_audit_script_is_non_gpu_and_dense_referenced():
    text = Path("scripts/run_current_goal_audit.sh").read_text()

    assert "randopt_lora_lab.goal_audit" in text
    assert "results/current_goal_audit_current" in text
    assert "--dense-confirmation-gate results/qproj_c2_vllm_shortlist_p64/shortlist_dense_confirmation/summary.json" in text
    assert "--search-quality-confirmation results/qproj_c2_vllm_shortlist_p64/search_quality_confirmation/summary.json" in text
    assert "MODE=confirm" not in text
    assert "experiments search" not in text
