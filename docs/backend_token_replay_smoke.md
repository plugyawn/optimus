# Backend Token Replay Smoke

Date: 2026-05-07

Artifact: `results/backend_token_replay_smoke`

Purpose: verify that `backend_token_replay_probe` works against a real vLLM backend with tokenized prompts, not decoded-string prefixes.

## Setup

- model: `Qwen/Qwen2.5-3B-Instruct`
- conditions: base and zero LoRA
- prompts: 1
- steps: 2
- prefix modes: `hf,vllm`
- top-k: 5
- vLLM: `0.20.1`
- hardware: 1x A6000 48GB
- vLLM eager mode: enabled

## Result

The smoke passed.

| Condition | Prefix mode | Rows | Top-1 equal rate | Mean top-k overlap | Max common abs logprob delta |
| --- | --- | ---: | ---: | ---: | ---: |
| base | hf | 2 | 1.0000 | 5.0000 | 0.49999 |
| base | vllm | 2 | 1.0000 | 5.0000 | 0.50000 |
| zero | hf | 2 | 1.0000 | 5.0000 | 0.49999 |
| zero | vllm | 2 | 1.0000 | 5.0000 | 0.50000 |

The generated token IDs matched at each checked step:

```text
step 0: 366  " <"
step 1: 9217 "answer"
```

The prompt contract also matched HF and vLLM unpadded prompt token hashes:

```text
40b268d3cde2c433b0edb250ad55505fb60b5f49c69e0438c0c6199adebf29c8
```

HF tokenizer padding side remained `left`; vLLM tokenizer padding side reported `right`.

## Interpretation

This proves the diagnostic can replay shared token-ID prefixes through vLLM. It does not prove backend parity for candidate LoRAs. The next useful run is the documented P64 token replay panel over the four previously divergent candidates.

Because the prompt IDs match and token replay works for base/zero, future candidate-level mismatches are more likely to be LoRA application/scaling, numerical logit drift, or generation-path amplification rather than decoded-prefix retokenization.
