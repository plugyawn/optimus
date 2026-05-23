# Backend Parity Gate P64 Tokenized vLLM

Date: 2026-05-07

Run: `results/backend_parity_gate_p64_tokenized_vllm`

Model: `Qwen/Qwen2.5-3B-Instruct`

Family: `factor_gaussian_lora`, rank 8, targets `q_proj,v_proj`, sigma `0.0075`, antithetic P=64.

Protocol: same PEFT reference as `results/backend_parity_gate_current_p64/peft`; new vLLM screen uses `--prompt-input token_ids`, 64 unique screen prompts, 8 unique holdout prompts, `max_new_tokens=32`, `stop_at_answer=true`, `promote=0`.

## Result

The gate still fails.

Tokenized prompt input fixes the base-score mismatch, and modestly improves ranking/output metrics, but it does not make vLLM a trusted selector.

| Metric | String-prompt vLLM | Token-ID vLLM |
| --- | ---: | ---: |
| Screen candidate/sec | 3.5452 | 2.0316 |
| Screen prompts/sec | not recorded here | 130.0215 |
| Base screen exact | 4.6875% | 6.2500% |
| PEFT base screen exact | 6.2500% | 6.2500% |
| Spearman vs PEFT | 0.6191 | 0.6572 |
| Pearson vs PEFT | 0.7931 | 0.8088 |
| Top-4 overlap | 4/4 | 4/4 |
| Top-8 overlap | 6/8 | 6/8 |
| Top-16 overlap | 10/16 | 11/16 |
| Mean abs score delta | 0.01953 | 0.01758 |
| Max abs score delta | 0.12500 | 0.12500 |
| Exact disagreement rate | 2.588% | 2.393% |
| Answer equal rate | 59.790% | 60.181% |
| Text equal rate | 52.661% | 52.905% |
| Max abs cap-hit delta by candidate | 0.953125 | 0.953125 |
| Max abs malformed delta by candidate | 0.703125 | 0.687500 |

Gate summary for the token-ID run:

| Gate | Result |
| --- | ---: |
| Protocol metadata | pass |
| Base rows present | pass |
| Adapter tensor parity | pass, 144 tensors checked from 1 adapter |
| Ranking correlation | fail |
| Output diff parity | fail |

The first attempted gate used `--adapter-sample 16`, but adapter tensor checking was CPU-bound for more than five minutes on the pod. It was stopped and rerun with `--adapter-sample 1`. This is enough for this prompt-input comparison because the same deterministic adapter materializer already passed the P64 full sample in the string-prompt gate, and prompt input mode does not alter adapter tensors.

## Prompt Contract

`vllm/prompt_contract.json` records `prompt_input=token_ids`.

Prompt hashes:

```text
screen:default  d3be917abe74f1c7efe10c2624e95efcfb10213db31db1d64e6050bf61b66cd8
holdout:default a8d7c68b5311f28ce56cf1a90a3afa7a6288381f7e76d7c4f01b3b7827374acf
```

vLLM sampling recorded:

```text
stop=["</answer>"]
include_stop_str_in_output=true
temperature=0.0
max_tokens=32
```

## Interpretation

The previous base mismatch was partly a prompt-input issue: tokenized vLLM prompts bring base screen exact from `4.6875%` to the PEFT value of `6.25%`.

The selector problem remains. The same PEFT-best candidate is still selected by vLLM, so there is no top-1 regret on this panel, but the score surface is still not close enough for vLLM-only quality claims:

- Spearman remains far below the `0.85` gate.
- Exact disagreement remains nonzero over 4096 common prompt/candidate rows.
- Large cap/malformed deltas remain, so sparse exact reward can still be backend-amplified.

Operational rule remains:

1. Use token-ID vLLM prompts for screening from now on.
2. Keep PEFT/HF confirmation for promoted candidates.
3. Do not claim vLLM-only selector parity from this backend.
4. Investigate LoRA/backend generation differences next, not prompt retokenization.

## Two-Stage Confirmation Path

The vLLM-only selector gate fails, but the proposal-plus-confirmation path passes
on this P64 panel:

```text
results/confirmation_economics_p64_tokenized_vllm
```

The trusted PEFT-best candidate was present at vLLM rank 1, so PEFT confirmation
of the top-1 proposal recovered zero regret. This is not enough for a general
quality claim, but it is a valid systems pattern for future searches:

```text
vLLM proposes a small top-K set
trusted PEFT/HF confirms that set
only confirmed PEFT/HF scores decide quality claims
```

Gate outcome:

| check | value |
| --- | ---: |
| best recovered at k | 1 |
| zero-regret k | 1 |
| eval-only speedup vs full PEFT screen | 10.55x |
| speedup including vLLM load/build but excluding PEFT load | 1.93x |
| confirmation gate | PASS |

This reframes the systems axis: vLLM does not need to be an exact selector if
its recall of PEFT-good candidates is high at small K and the trusted
confirmation cost stays below the full PEFT screen cost.

## Next Probe

The next useful probe should focus on candidate-level generation divergence under token-ID prompts:

- take the top PEFT candidate, the top vLLM candidate, and the worst output-diff candidates;
- run full token-ID rollouts with PEFT and vLLM on 16-32 prompts;
- log first divergence token, exact answer, cap/malformed, and finish reason;
- test whether divergence is reduced by eager mode, lower `max_loras`, or one-adapter-at-a-time serving.

If those do not reduce deltas, the practical systems path is vLLM proposal plus PEFT confirmation, not vLLM as the final selector.
