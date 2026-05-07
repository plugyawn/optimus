# Backend Contract Smoke

Date: 2026-05-07

Artifact: `results/backend_contract_smoke`

Purpose: verify that backend probes now save a `prompt_contract.json` artifact on a real vLLM backend before generation.

## Setup

- model: `Qwen/Qwen2.5-3B-Instruct`
- backend probe: `backend_rollout_probe`
- prompts: 1
- conditions: base and zero LoRA
- `max_new_tokens`: 4
- `stop_at_answer`: true
- hardware: 1x A6000 48GB
- vLLM: `0.20.1`

## Contract Result

The artifact was written successfully and includes both HF and best-effort vLLM tokenizer contracts.

| Field | HF | vLLM |
| --- | --- | --- |
| tokenizer class | `Qwen2TokenizerFast` | `CachedQwen2TokenizerFast` |
| padding side | `left` | `right` |
| pad token id | `151643` | `151643` |
| eos token id | `151645` | `151645` |
| answer stop ids | `[522, 9217, 29]` | `[522, 9217, 29]` |
| prompt token hash | `40b268d3cde2c433b0edb250ad55505fb60b5f49c69e0438c0c6199adebf29c8` | same |

vLLM sampling settings were also captured:

```json
{
  "max_tokens": 4,
  "temperature": 0.0,
  "stop": ["</answer>"],
  "include_stop_str_in_output": true,
  "ignore_eos": false,
  "skip_special_tokens": true
}
```

## Interpretation

This rules out raw prompt-token mismatch for this single-prompt smoke. It does not explain the P64 LoRA rollout divergence.

The padding-side difference is worth keeping in the contract: it did not alter unpadded prompt IDs here, but it can matter for batched HF replay, stopping criteria, and any future probe that compares padded positions rather than generated-token continuations.

Next backend diagnosis should compare logits along a shared token-ID prefix, not just decoded string prefixes.
