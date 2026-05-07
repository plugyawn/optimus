# Backend Step And Rollout Parity P64

Date: 2026-05-07

Model: `Qwen/Qwen2.5-3B-Instruct`

Family: `factor_gaussian_lora`, rank 8, targets `q_proj,v_proj`, sigma `0.0075`.

Artifacts:

- `results/backend_step_parity_probe_p64`
- `results/backend_step_parity_probe_p64_vllm_prefix`
- `results/backend_rollout_probe_p64_token_ids`

## Why This Probe Exists

The P64 backend gate showed adapter tensor parity, strong vLLM speed, and same top-4 overlap, but it failed strict ranking/output parity:

- Spearman: `0.6191`, below the `0.85` gate.
- Eval-only speedup: `22.3x`.
- vLLM selected the same best candidate under the PEFT trusted score, but broader ranking drift remained.

This follow-up separates three possible causes:

1. different adapter tensors;
2. different next-token model distributions;
3. generation-path amplification from small/token-boundary differences.

Adapter tensor parity had already passed, so the remaining question is whether vLLM and PEFT/HF disagree during generation in a way that can affect sparse exact reward.

## Stepwise Next-Token Probe

The stepwise probe compares PEFT/HF and vLLM top-k next-token distributions along generated prefixes.

| Prefix mode | Rows | Overall top-1 equal rate |
| --- | ---: | ---: |
| HF prefixes | 395 | 0.9848 |
| vLLM prefixes | 401 | 0.9875 |

Per-condition summary:

| Condition | HF-prefix top-1 equal | vLLM-prefix top-1 equal |
| --- | ---: | ---: |
| base | 0.9841 | 0.9848 |
| zero LoRA | 0.9841 | 0.9848 |
| `seed1411240924:s0.0075:sign1` | 0.9726 | 0.9857 |
| `seed1632697641:s0.0075:sign-1` | 0.9825 | 0.9825 |
| `seed221822464:s0.0075:sign-1` | 1.0000 | 1.0000 |
| `seed326653716:s0.0075:sign1` | 0.9863 | 0.9868 |

Qualitatively, several mismatches are near-ties or token-boundary alternatives:

- base/zero example `782`, step 4: `'9'` vs `'2'`, both around tied logprob.
- best candidate example `238`, step 8: `' +'` vs `'+'`.
- best candidate example `238`, step 17: `' </'` vs `'</'`.
- `seed1632697641` example `134`, step 12: `'>'` vs `'></'`.

The stepwise result is therefore not "vLLM math is wildly wrong." It is closer to "small logit/tokenization/stop-boundary differences are rare per token, but sparse exact reward can amplify them."

## Token-ID Rollout Probe

The rollout probe records token IDs from full PEFT/HF and vLLM generations on a 4-prompt panel.

Overall:

| Metric | Value |
| --- | ---: |
| Rows | 24 |
| Exact equal rate | 1.0000 |
| Answer equal rate | 0.5833 |
| Text equal rate | 0.5000 |
| `max_new_tokens` | 32 |

Per-condition token agreement:

| Condition | Token-ID equal rate | First divergence rate | Mean first divergence | Text equal | Answer equal | Exact equal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 0.7500 | 0.2500 | 11.00 | 0.7500 | 0.7500 | 1.0000 |
| zero LoRA | 0.7500 | 0.2500 | 11.00 | 0.7500 | 0.7500 | 1.0000 |
| `seed1411240924:s0.0075:sign1` | 0.0000 | 1.0000 | 14.00 | 0.5000 | 0.5000 | 1.0000 |
| `seed1632697641:s0.0075:sign-1` | 0.2500 | 0.7500 | 4.00 | 0.5000 | 0.5000 | 1.0000 |
| `seed221822464:s0.0075:sign-1` | 0.0000 | 1.0000 | 13.75 | 0.5000 | 0.7500 | 1.0000 |
| `seed326653716:s0.0075:sign1` | 0.0000 | 1.0000 | 7.50 | 0.0000 | 0.2500 | 1.0000 |

The 4-prompt rollout panel was too small to reproduce exact-score disagreement, but it did confirm real token-path divergence. Examples:

- base/zero example `238`: PEFT emits `35*27+25-9`; vLLM emits `35*27+25*9`.
- `seed1411240924` example `238`: PEFT emits `35*25+27*9`; vLLM emits malformed `3527+9=367`.
- `seed1632697641` examples `238` and `999`: PEFT starts reasoning text and hits the cap/malformed path; vLLM emits answer tags immediately.
- `seed326653716` examples `134`, `238`, and `782`: PEFT starts reasoning text and caps/malforms; vLLM emits compact answer tags.

## Interpretation

The backend problem is now narrower:

- adapter tensor materialization is not the blocker;
- next-token top-1 parity is high but not exact;
- full generation can diverge at token 0 or soon after for LoRA candidates;
- sparse exact reward and cap/malformed outcomes amplify those divergences into ranking drift.

This means the vLLM path is useful as a systems accelerator, but not yet as a sole quality oracle. The current safe operating rule remains:

1. Use vLLM for high-throughput screening.
2. Promote top-K candidates to PEFT/HF confirmation.
3. Make quality claims only from PEFT/HF-confirmed candidates and heldout rows.
4. Treat prompt wording, stop criteria, and token cap as experimental variables, not cleanup knobs.

## Next Checks

The highest-leverage backend fixes are:

1. Compare exact prompt token IDs and stop strings before generation for PEFT/HF and vLLM.
2. Run `backend_token_replay_probe`, which feeds the same generated token-ID prefix into both engines and compares logits before sampling the next token.
3. Audit whether vLLM chat-template, stop-token, EOS, or added-special-token handling differs from the HF path.
4. Test whether answer-only output with stricter stop handling reduces backend divergence without changing the task semantics.

New backend probes should save `prompt_contract.json` before generation. That file must include exact HF prompt token IDs, best-effort exact vLLM prompt token IDs, tokenizer special-token IDs, answer-stop token IDs, and the requested/actual vLLM sampling stop settings. Without that artifact, a generation-parity failure cannot be cleanly separated from prompt/tokenizer/stop-contract drift.

Token-ID replay command template:

```bash
python -m randopt_lora_lab.backend_token_replay_probe \
  --out results/backend_token_replay_probe_p64 \
  --data data/countdown_generated_1200_seed20260507.json \
  --prompts 4 \
  --seed 4242 \
  --rank 8 \
  --targets q_proj,v_proj \
  --top-k 20 \
  --max-steps 24 \
  --prefix-modes hf,vllm \
  --include-zero \
  --stop-at-answer \
  --candidate factor_gaussian_lora:seed1411240924:s0.0075:sign1 \
  --candidate factor_gaussian_lora:seed1632697641:s0.0075:sign-1 \
  --candidate factor_gaussian_lora:seed326653716:s0.0075:sign1 \
  --candidate factor_gaussian_lora:seed221822464:s0.0075:sign-1
```

Until those pass, the search pipeline should be vLLM-screen plus PEFT-confirm, not vLLM-only search.
