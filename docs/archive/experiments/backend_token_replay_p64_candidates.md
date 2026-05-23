# Backend Token Replay P64 Candidates

Date: 2026-05-07

Artifact: `results/backend_token_replay_p64_candidates`

Purpose: replay the same token-ID prefixes through PEFT/HF and vLLM for the four P64 candidates that previously showed rollout divergence. This removes decoded-string prefix retokenization from the diagnostic path.

## Setup

- model: `Qwen/Qwen2.5-3B-Instruct`
- family: `factor_gaussian_lora`
- rank: `8`
- targets: `q_proj,v_proj`
- candidate sigma: `0.0075`
- prompts: `4`
- max steps: `18`
- top-k compared per step: `20`
- prefix input: token IDs
- prefix modes: `hf,vllm`
- vLLM: `0.20.1`
- hardware: 1x A6000 48GB
- vLLM eager mode: disabled

## Prompt Contract

`prompt_contract.json` records matching HF and vLLM unpadded prompt-token hashes:

```text
16bfaedd2860940716b90266c41173236e4c68131fd9265418c7f8f1e1d3100a
```

Tokenizer and stop details:

- HF tokenizer: `Qwen2TokenizerFast`, padding side `left`
- vLLM tokenizer: `CachedQwen2TokenizerFast`, padding side `right`
- answer stop text: `</answer>`
- answer stop IDs: `[522, 9217, 29]`
- vLLM sampling: `max_tokens=1`, `temperature=0.0`, `stop=["</answer>"]`, `include_stop_str_in_output=true`

The padding-side difference is not a prompt-content difference here because the replay passes unpadded token-ID prompts.

## Result

Overall top-1 equality across all checked next-token rows was `0.9975`.

| Condition | Prefix mode | Rows | Top-1 equal | First mismatch rate | Mean first mismatch step | Mean top-k overlap |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| base | hf | 66 | 1.0000 | 0.0000 | - | 19.4394 |
| base | vllm | 66 | 1.0000 | 0.0000 | - | 19.4697 |
| zero LoRA | hf | 66 | 1.0000 | 0.0000 | - | 19.4394 |
| zero LoRA | vllm | 66 | 1.0000 | 0.0000 | - | 19.4697 |
| `seed1411240924:s0.0075:sign1` | hf | 71 | 1.0000 | 0.0000 | - | 19.3803 |
| `seed1411240924:s0.0075:sign1` | vllm | 71 | 1.0000 | 0.0000 | - | 19.3944 |
| `seed1632697641:s0.0075:sign-1` | hf | 57 | 1.0000 | 0.0000 | - | 19.4561 |
| `seed1632697641:s0.0075:sign-1` | vllm | 57 | 1.0000 | 0.0000 | - | 19.4386 |
| `seed221822464:s0.0075:sign-1` | hf | 72 | 1.0000 | 0.0000 | - | 19.4167 |
| `seed221822464:s0.0075:sign-1` | vllm | 72 | 1.0000 | 0.0000 | - | 19.4583 |
| `seed326653716:s0.0075:sign1` | hf | 69 | 0.9855 | 0.2500 | 0.0000 | 19.4783 |
| `seed326653716:s0.0075:sign1` | vllm | 67 | 0.9851 | 0.2500 | 0.0000 | 19.4030 |

Only two rows mismatched, both on the same candidate/example at step 0:

```text
condition: seed326653716:s0.0075:sign1
example: 238
prefix mode: hf
step: 0
HF top-1:   2014 " To"
vLLM top-1:  366 " <"
top-k overlap: 19/20
max common abs logprob delta: 0.51795

condition: seed326653716:s0.0075:sign1
example: 238
prefix mode: vllm
step: 0
HF top-1:   2014 " To"
vLLM top-1:  366 " <"
top-k overlap: 19/20
max common abs logprob delta: 0.46309
```

## Interpretation

This clears the largest suspected confound in the earlier stepwise probe: decoded-string prefix retokenization. When the engines are fed shared token-ID prefixes, base and zero LoRA are exact on top-1 choice across the panel, and three of four nonzero candidates are exact on top-1 choice across all checked rows.

The remaining mismatch is candidate-specific and happens immediately from the original prompt, so it is not caused by accumulated generation-path drift. The top-k overlap is still high at `19/20`, which points to a small backend/LoRA/numeric difference near the decision boundary rather than a gross adapter export or prompt-contract error.

This does not yet justify vLLM-only quality claims, because the prior P64 rank correlation gate still failed and sparse exact reward can amplify even rare token choices. It does justify tightening the backend rule:

1. vLLM screening is operationally valid for high-throughput candidate search.
2. Candidate promotion still needs PEFT/HF confirmation until ranking parity passes on the actual screen distribution.
3. Future backend parity gates should use token-ID replay, not decoded-prefix replay, before attributing disagreement to LoRA application.
4. Full generation parity should be remeasured after switching the rollout path to token-ID prompts where possible.

## Next Gate

Run one full P64 screen with tokenized vLLM prompts and the same prompt/stop contract, then compare:

- vLLM screen speed against the prior naive PEFT/HF path;
- top-K overlap against PEFT/HF confirmation;
- Spearman rank correlation;
- exact reward agreement by candidate;
- probed rollouts for the promoted candidates.

If tokenized prompts lift ranking parity above the backend gate, vLLM can become the primary screen backend. If rank drift remains, keep vLLM as a fast proposal engine and PEFT/HF as the trusted selector.
