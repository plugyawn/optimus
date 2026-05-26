# Kernel Experiment Summary

| id | date | hardware | change | validation | result |
| --- | --- | --- | --- | --- | --- |
| bridge-cache-p128 | 2026-05-26 | A100-SXM4-80GB | Cached vLLM LoRA-kernel factor stacks and row metadata in the true-lazy bridge | `222 passed`; p128 adapter/lazy replay; plot validator | p128 lazy bridge reached `1.310 cand/s` at rank64 vs native subspace-as-LoRA `1.162 cand/s`; best score matched `25/128`; generation-score parity close but not exact |
| strict-signature-p1 | 2026-05-26 | L40S-48GB | Added staged one-token adapter-vs-lazy signature probe with prompt-tail top-logprobs | zero-scale pass; nonzero torch/vLLM-kernel fail under `1e-2` top-logprob tolerance | generated token parity is `2/2`, but nonzero logprob parity is not closed: torch lazy max gen/top diffs `0.018/0.153`, vLLM-kernel bridge `0.080/0.250`; fp32 lazy compute did not help |
