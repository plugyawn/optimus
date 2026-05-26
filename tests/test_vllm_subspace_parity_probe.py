from __future__ import annotations

from types import SimpleNamespace

from scripts import probe_vllm_subspace_parity as probe


class _LogProb:
    def __init__(self, logprob: float, decoded_token: str | None = None) -> None:
        self.logprob = logprob
        self.decoded_token = decoded_token


def _output(token_id: int, logprob: float, *, prompt_tail: list[dict[int, _LogProb]] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_token_ids=[11, 12, 13],
        prompt_logprobs=[None, *(prompt_tail or [{12: _LogProb(-0.5)}, {13: _LogProb(-0.25)}])],
        outputs=[
            SimpleNamespace(
                token_ids=[token_id],
                logprobs=[{token_id: _LogProb(logprob, "x"), 99: _LogProb(-3.0, "y")}],
            )
        ],
    )


def test_signature_rows_extract_generated_and_prompt_tail_logprobs() -> None:
    rows = probe._signature_rows(
        [_output(42, -0.125)],
        candidate_id="seed1:+:r64:rho0.4",
        examples=[SimpleNamespace(id=7)],
        backend="adapter",
        prompt_tail_tokens=2,
    )

    assert rows[0]["generated"]["token_id"] == 42
    assert rows[0]["generated"]["logprob"] == -0.125
    assert rows[0]["generated"]["top_logprobs"]["42"]["token"] == "x"
    assert [item["token_id"] for item in rows[0]["prompt_tail"]] == [12, 13]


def test_compare_rows_reports_token_and_logprob_parity() -> None:
    adapter = probe._signature_rows(
        [_output(42, -0.125)],
        candidate_id="seed1:+:r64:rho0.4",
        examples=[SimpleNamespace(id=7)],
        backend="adapter",
        prompt_tail_tokens=2,
    )
    lazy = probe._signature_rows(
        [_output(42, -0.13)],
        candidate_id="seed1:+:r64:rho0.4",
        examples=[SimpleNamespace(id=7)],
        backend="lazy",
        prompt_tail_tokens=2,
    )

    rows, summary = probe._compare_rows(adapter, lazy)

    assert rows[0]["generated_token_match"] is True
    assert rows[0]["generated_logprob_abs_diff"] == abs(-0.125 + 0.13)
    assert summary["generated_token_match_rate"] == 1.0
    assert summary["max_common_top_logprob_abs_diff"] == abs(-0.125 + 0.13)


def test_compare_rows_flags_generated_token_mismatch() -> None:
    adapter = probe._signature_rows(
        [_output(42, -0.125)],
        candidate_id="seed1:+:r64:rho0.4",
        examples=[SimpleNamespace(id=7)],
        backend="adapter",
        prompt_tail_tokens=1,
    )
    lazy = probe._signature_rows(
        [_output(43, -0.13)],
        candidate_id="seed1:+:r64:rho0.4",
        examples=[SimpleNamespace(id=7)],
        backend="lazy",
        prompt_tail_tokens=1,
    )

    rows, summary = probe._compare_rows(adapter, lazy)

    assert rows[0]["generated_token_match"] is False
    assert summary["generated_token_match_count"] == 0
