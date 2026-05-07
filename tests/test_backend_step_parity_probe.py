from randopt_lora_lab.backend_step_parity_probe import next_prefix_token, summarize


def test_next_prefix_token_modes():
    assert next_prefix_token("hf", "A", "B", False) == "A"
    assert next_prefix_token("vllm", "A", "B", False) == "B"
    assert next_prefix_token("match", "A", "A", True) == "A"
    assert next_prefix_token("match", "A", "B", False) is None


def test_summarize_records_first_mismatch():
    rows = [
        {
            "condition": "base",
            "prefix_mode": "hf",
            "example_id": "a",
            "step": 0,
            "top1_equal": True,
            "topk_overlap": 5,
            "hf_top1_token": "x",
            "vllm_generated_token": "x",
            "max_common_abs_logprob_delta": 0.1,
        },
        {
            "condition": "base",
            "prefix_mode": "hf",
            "example_id": "a",
            "step": 1,
            "top1_equal": False,
            "topk_overlap": 4,
            "hf_top1_token": "y",
            "vllm_generated_token": "z",
            "max_common_abs_logprob_delta": 0.3,
        },
        {
            "condition": "base",
            "prefix_mode": "hf",
            "example_id": "b",
            "step": 0,
            "top1_equal": True,
            "topk_overlap": 5,
            "hf_top1_token": "x",
            "vllm_generated_token": "x",
            "max_common_abs_logprob_delta": 0.2,
        },
    ]
    summary = summarize(rows)
    group = summary["conditions"]["base|hf"]
    assert summary["overall_top1_equal_rate"] == 2 / 3
    assert group["first_mismatch_rate"] == 0.5
    assert group["mean_first_mismatch_step"] == 1
    assert group["max_common_abs_logprob_delta"] == 0.3
