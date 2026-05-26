from __future__ import annotations

from types import SimpleNamespace

import torch

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


def test_compare_capture_rows_reports_layer_drift() -> None:
    adapter = [
        {
            "candidate_id": "seed1:+:r64:rho0.4",
            "target_id": "layer_0.self_attn.qkv_proj",
            "call_index": 0,
            "layer_index": 0,
            "suffix": "qkv_proj",
            "tensor": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        }
    ]
    lazy = [
        {
            "candidate_id": "seed1:+:r64:rho0.4",
            "target_id": "layer_0.self_attn.qkv_proj",
            "call_index": 0,
            "layer_index": 0,
            "suffix": "qkv_proj",
            "tensor": torch.tensor([[1.0, 1.5], [2.5, 4.0]]),
        }
    ]

    rows, summary = probe._compare_capture_rows(adapter, lazy)

    assert rows[0]["shape_match"] is True
    assert rows[0]["max_abs"] == 0.5
    assert summary["comparisons"] == 1
    assert summary["max_abs"] == 0.5
    assert summary["worst"]["target_id"] == "layer_0.self_attn.qkv_proj"


def test_compare_capture_rows_tracks_missing_adapter_rows() -> None:
    rows, summary = probe._compare_capture_rows(
        [],
        [
            {
                "candidate_id": "seed1:+:r64:rho0.4",
                "target_id": "layer_0.self_attn.qkv_proj",
                "call_index": 0,
                "layer_index": 0,
                "suffix": "qkv_proj",
                "tensor": torch.zeros(1, 2),
            }
        ],
    )

    assert rows == []
    assert summary["missing_adapter_count"] == 1
    assert summary["comparisons"] == 0
