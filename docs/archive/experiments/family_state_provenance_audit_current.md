# Family State Provenance Audit

## Verdict

The current activation-spectral q-only c2 P64 confirmations are not valid adapter-identity confirmations.

Both runs have `vllm/family_state.pt`, but neither confirmed PEFT run copied or recorded that state. Since `activation_spectral_lora*` directions depend on the activation basis, the candidate key alone is not enough to identify the adapter direction.

This means the existing confirmations are useful as rough evidence about the family, but they should not be used for strict vLLM-to-PEFT selector parity or dense-regret claims.

## Command

```bash
PYTHONPATH=. python -m randopt_lora_lab.family_state_provenance_audit \
  --results-root results \
  --out results/family_state_provenance_audit_current \
  --no-fail
```

## Result

```text
Gate: FAIL

failed runs:
  results/qproj_c2_vllm_shortlist_p64
  results/qproj_c2_vllm_shortlist_p64_seed20260508
```

For both failed runs:

```text
confirmed_summary_family_state_file_present: false
confirmed_family_state_present: false
confirmed_family_state_matches_vllm: false
confirmed_family_state_summary_present: false
confirmed_family_state_summary_loaded: false
confirmed_family_state_summary_points_to_state: false
```

The vLLM state hashes were:

```text
qproj_c2_vllm_shortlist_p64:
  cdf619d779c3d092a4520a78ba28353471809641a64e0ef3e4d0215361e532f1

qproj_c2_vllm_shortlist_p64_seed20260508:
  6e1cb42fb261d6d3128b200e56d678de7346660443f7ef54ac89bc82267ee235
```

## Fix Landed

`experiments search` and `experiments halving` now accept:

```bash
--family-state-file path/to/family_state.pt
```

`scripts/run_vllm_shortlist_confirmation.sh` now auto-passes:

```text
$OUT_ROOT/vllm/family_state.pt
```

to PEFT confirmation when it exists, records a copied `confirmed/family_state.pt`, and runs this provenance audit by default.

## Consequence

The next GPU run should not scale population. It should first rerun the small PEFT confirmation for the selected q-only c2 shortlist using the saved vLLM basis, then regenerate:

```text
confirmed/summary.json
confirmed/family_state.pt
confirmed/family_state_summary.json
family_state_provenance_audit/summary.json
shortlist_dense_confirmation/summary.json
search_quality_confirmation/summary.json
```

Only after that passes should q-only c2 speed/quality claims be considered live
again. The preferred path is now a fresh corrected confirmation, not replaying
the old `default,reordered,xml` source panel:

```bash
scripts/run_qproj_c2_corrected_confirmation.sh
```

When a GPU is intentionally available:

```bash
MODE=confirm scripts/run_qproj_c2_corrected_confirmation.sh
```

This creates a fresh vLLM screen with `default,reordered`, requires all requested
prompt variants to be base-valid, writes dense-referenced confirmation,
search-quality, score-sanity, provenance, and current goal-audit artifacts.

The exact replay wrapper below is now primarily forensic: it verifies whether an
old source panel was confirmed with the same saved vLLM family state. It should
not be used as the primary prompt-agnostic quality claim if the source vLLM
summary still contains base-invalid XML stress prompts.

Use the existing-panel replay wrapper so the source run is not mutated:

```bash
SOURCE_ROOT=results/qproj_c2_vllm_shortlist_p64 \
OUT_ROOT=results/qproj_c2_vllm_shortlist_p64_default_exact_k4 \
FAMILY=activation_spectral_lora_c2 \
TARGETS=q_proj \
SEED=20260507 \
DATA=data/countdown_generated_1200_seed20260507.json \
SHORTLIST_POLICY=default_exact \
SHORTLIST_K=4 \
CONFIRM_KS=1,2,4 \
CONFIRM_MAX_K=4 \
CONFIRM_MAX_DENSE_REGRET=0.015625 \
scripts/run_existing_vllm_shortlist_confirmation.sh
```

The guarded exact-replay form defaults to preflight-only and should be used
before attaching a GPU:

```bash
scripts/run_qproj_c2_exact_replay.sh
```

When a GPU is intentionally available for forensic replay:

```bash
MODE=confirm scripts/run_qproj_c2_exact_replay.sh
```

Confirmation mode also writes a current completion audit to:

```text
results/qproj_c2_vllm_shortlist_p64_default_exact_k4/current_goal_audit
```

Every guarded replay also writes a forensic manifest to:

```text
results/qproj_c2_vllm_shortlist_p64_default_exact_k4/replay_manifest
```

The manifest is diagnostic, not a substitute for the completion audit. It records
which dense/vLLM/shortlist/score-sanity/confirmation/provenance/audit artifacts
are present, which gates failed, and whether the chained goal audit passed. The
score-sanity audit checks top-candidate cap hits, malformed rates, answer closure,
screen sample size, and whether the top candidate clears the base screen score.

Do not use `xml` as a default vLLM screening prompt variant. Existing artifacts
show `default` and `reordered` are base-healthy, while `xml` is consistently
malformed enough to fail the score-sanity gate. XML can still be used as an
explicit stress prompt, but prompt-agnostic quality claims should start from
`default,reordered` and require all requested variants to be base-valid.

That wrapper copies `SOURCE_ROOT/dense` and `SOURCE_ROOT/vllm` into `OUT_ROOT`,
writes a selector-union shortlist, then runs PEFT confirmation with:

```text
--family-state-file "$OUT_ROOT/vllm/family_state.pt"
```

The wrapper also supports a local preflight that does not load the model:

```bash
PREFLIGHT_ONLY=1 \
SOURCE_ROOT=results/qproj_c2_vllm_shortlist_p64 \
OUT_ROOT=/tmp/qproj_c2_default_exact_k4_preflight \
FAMILY=activation_spectral_lora_c2 \
TARGETS=q_proj \
SHORTLIST_POLICY=default_exact \
SHORTLIST_K=4 \
scripts/run_existing_vllm_shortlist_confirmation.sh
```

Current preflight result:

```text
pass: true
shortlist_rows: 4
shortlist_policy: default_exact
vllm_candidates: 64
source_family_state_sha256:
  cdf619d779c3d092a4520a78ba28353471809641a64e0ef3e4d0215361e532f1
vllm_family_state_sha256:
  cdf619d779c3d092a4520a78ba28353471809641a64e0ef3e4d0215361e532f1
```

After PEFT confirmation, validity and provenance gates are non-fatal inside
the wrapper: they still write their `summary.json` files, and downstream dense
confirmation, search-quality confirmation, and current goal audit still run.
This is intentional, because negative or invalid runs must preserve forensic
evidence instead of stopping at the first red gate.
