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
