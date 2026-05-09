# Corrected QProj C2 Dense-Referenced Multi-Run A100

## Runs

```text
date: 2026-05-09
pod: randopt-qproj-corrected-0509, Prime pod 5a9d51477dd5449fb68c15030657a6b1
gpu: 1x NVIDIA A100-SXM4-80GB
pod status after sync: terminated
model: Qwen/Qwen2.5-3B-Instruct
proposal family: activation_spectral_lora_c2
targets: q_proj
population: 64
screen prompts: 64
holdout prompts: 128 for PEFT dense/confirmed, 8 for vLLM proposal
prompt variants: default,reordered
shortlist policy: default_exact
confirm ks: 1,2,4
```

Two corrected confirmations were run:

```text
results/qproj_c2_vllm_shortlist_p64_default_reordered
results/qproj_c2_vllm_shortlist_p64_default_reordered_seed20260510
```

Compact evidence was synced locally under:

```text
results/qproj_corrected_seed20260507_evidence.tar.zst
results/qproj_corrected_seed20260510_evidence.tar.zst
results/qproj_corrected_adapter_manifests.tar.zst
```

The materialized vLLM adapter directories were not copied before shutdown. Each
was about 578 MB and the direct pod-to-local link was throttled badly. The
compact evidence, run summaries, adapter manifests, and family-state summaries
were preserved; full adapter blobs would need to be regenerated from the run
configuration if needed.

## Corrected Multi-Run Gate

The new aggregate gate is:

```text
python -m randopt_lora_lab.dense_referenced_multirun_gate \
  --run results/evidence_unpack_seed20260507/qproj_c2_vllm_shortlist_p64_default_reordered \
  --run results/evidence_unpack_seed20260510/qproj_c2_vllm_shortlist_p64_default_reordered_seed20260510 \
  --out results/qproj_c2_dense_referenced_multirun_gate
```

Result:

```text
gate: PASS
runs: 2
validity pass count: 2
shortlist dense pass count: 2
search quality pass count: 2
score sanity pass count: 2
family-state provenance pass count: 2
artifact complete count: 2
prompt robust count: 2
minimum strict-holdout delta vs dense best: +1.5625 pp
mean strict-holdout delta vs dense best: +2.734375 pp
minimum full speedup: 5.9847x
mean full speedup: 7.9336x
```

Per-run highlights:

| seed | zero dense-regret k | dense best recovered k | strict delta vs dense best | full speedup |
| ---: | ---: | ---: | ---: | ---: |
| 20260507 | 4 | 4 | +1.5625 pp | 5.9847x |
| 20260510 | 1 | null | +3.90625 pp | 9.8825x |

The second run did not recover the exact dense-best seed/spec, but it did
recover a zero-regret-or-better shortlist candidate by k=1 and beat dense best
on strict holdout.

## Top-Level Audit Status

After wiring the corrected dense-referenced multi-run route into
`goal_audit.py`, the current local audit is:

```text
out: results/current_goal_audit_qproj_corrected_local
pass: false
failed: official full-Gaussian baseline validity
```

The corrected route now satisfies the local goal-audit checks for:

```text
quality parity
stability parity
speed parity
accelerated evaluation route
adapter identity provenance
multi-run prompt-robust confirmation
prompt robustness
drift parity
eval validity
score sanity
adapter convenience
```

The remaining red gate is deliberately not weakened: the old paper-style
full-Gaussian reproduction artifact still fails the official-baseline metadata
check. That is separate from the corrected q-proj two-run evidence.

## Interpretation

This is a stronger positive than the earlier default,reordered,xml selector
runs. The corrected `default,reordered` route no longer fails the dense-regret
gate, and it produces repeated strict-holdout wins over dense with useful
end-to-end speedup.

The claim should still be phrased carefully:

```text
Supported:
  corrected q-proj activation-spectral LoRA can be used as a fast proposal
  family, then PEFT-confirmed, with repeated dense-referenced strict-holdout
  wins at P64 and about 6x-10x full speedup.

Not yet supported:
  a paper-style official dense Gaussian reproduction claim;
  a Spearman-style dense ranking surrogate claim;
  preservation of full materialized vLLM adapter blobs from this pod.
```

The next experimental gate should either rerun the official dense baseline with
current metadata or run a larger repeated corrected-route panel. It should not
go back to the old robust-mean prompt selector.
