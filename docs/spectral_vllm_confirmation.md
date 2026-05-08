# Spectral vLLM Confirmation Gate

## Purpose

This is the next falsification run for the accelerated-search path. It tests two
claims separately:

```text
quality/stability: calibrated spectral LoRA can compete with dense Gaussian
systems: vLLM can cheaply propose spectral candidates for PEFT confirmation
```

The two claims must not be merged. vLLM-only selector parity is still a strict
backend gate and remains separate from this proposal-plus-confirmation route.

## Entry Point

```bash
OUT_ROOT=results/spectral_vllm_confirmation_rank32_c1p5_p64 \
FAMILY=spectral_projected_gaussian_rank_r_c1p5 \
RANK=32 \
POPULATION=64 \
PROMPTS=64 \
HOLDOUT_PROMPTS=256 \
SIGMA_VALUES=0.0005,0.001,0.002 \
scripts/run_spectral_vllm_confirmation.sh
```

The script produces:

```text
$OUT_ROOT/dense
$OUT_ROOT/control
$OUT_ROOT/spectral
$OUT_ROOT/parity
$OUT_ROOT/vllm_spectral
$OUT_ROOT/confirmation
$OUT_ROOT/dense_reference_confirmation
```

`$OUT_ROOT/parity` is the quality/stability artifact. `$OUT_ROOT/confirmation`
is the same-family systems artifact. `$OUT_ROOT/dense_reference_confirmation`
checks whether the vLLM shortlist plus spectral PEFT confirmation also recovers
the dense Gaussian reference best; without it, a confirmation pass is not dense
quality evidence.

## Defaults

The default candidate family is:

```text
spectral_projected_gaussian_rank_r_c1p5
```

That choice comes from the P16 spectral calibration smoke, where rank-32
spectral scales were the only projected family variants that looked worth
testing under an accelerated adapter backend. It is not a proven winner.

The vLLM proposal path defaults to:

```text
prompt_input=token_ids
prompt_variants=default,reordered,xml
score_mode=robust_mean
min_selection_prompt_variants=2
proposal_score_col=selection_score
```

If the base model fails the prompt-validity thresholds on fewer than two prompt
variants, the vLLM proposal run should abort. That is a useful negative result,
not a script failure.

## Gate Reading

Treat the run as positive only if all of these hold:

```text
1. $OUT_ROOT/dense/validity, control/validity, and spectral/validity pass.
2. $OUT_ROOT/parity shows spectral selected-regret near zero and no worse
   stability than the matched control arm.
3. spectral holdout lift is not just a one-example tie versus dense/control.
4. $OUT_ROOT/confirmation recovers the PEFT spectral best within CONFIRM_MAX_K.
5. confirmation speedup remains positive after vLLM load/build and PEFT
   confirmation cost.
6. $OUT_ROOT/dense_reference_confirmation recovers the dense Gaussian best
   within DENSE_REF_MAX_K with negligible dense regret.
7. cap-hit and malformed metrics do not regress on the promoted candidates.
```

A successful confirmation gate only says vLLM is a fast proposal engine for this
same spectral family. It does not prove that vLLM itself is an exact quality
selector, and it does not prove that spectral LoRA beats dense Gaussian.
Dense-reference confirmation is the extra check that prevents same-family
spectral recall from being mistaken for dense RandOpt parity.

## Negative Outcomes

Interpret failures directly:

```text
parity fails, confirmation passes:
  systems path is useful, but the spectral family is not yet a quality answer.

parity passes, confirmation fails:
  family may be useful, but vLLM proposal scoring is not reliable enough.

confirmation passes, dense-reference confirmation fails:
  vLLM can recover the spectral PEFT best, but the spectral family is still not
  a dense RandOpt replacement.

both fail:
  do not scale this spectral configuration.

validity fails:
  no quality claim; rerun only after fixing prompt/cap/parser issues.
```

The first scale-up should require multi-seed replication. A single P64 pass is
only permission for a larger diagnostic, not a project-level claim.

Use the multi-run gate before interpreting repeated runs:

```bash
OUT_ROOT=results/spectral_vllm_multirun_rank32_c1p5_p64 \
RUN_SEEDS=20260507,20260508 \
POPULATION=64 \
PROMPTS=64 \
HOLDOUT_PROMPTS=256 \
VLLM_HOLDOUT_PROMPTS=8 \
scripts/run_spectral_vllm_multirun_gate.sh
```

This gate keeps same-family systems confirmation, dense-referenced
confirmation, dense parity, validity, and prompt robustness separate. It should
fail a default-prompt-only run even if vLLM shortlisting is fast.

The wrapper sets `RUN_VLLM_FIRST=1` by default. That makes the robust vLLM
prompt-validity check run before the expensive PEFT dense/control/spectral arms,
so a prompt-protocol failure is cheap.

## P16 A100 Update

The first A100 P16 run is recorded in
`docs/spectral_vllm_confirmation_p16_a100.md` and
`results/spectral_vllm_confirmation_rank32_c1p5_p16_default`.

It found a systems positive but quality negative:

```text
confirmation gate: pass
zero-regret k: 2
eval-only speedup at k=2: 26.39x
full-without-PEFT-load speedup at k=2: 8.87x
spectral dense-parity gate: fail
spectral Spearman vs dense: 0.300
spectral selected regret: 1.5625 percentage points
```

The old default `ENSEMBLE_KS=1,4,8,16` made a P16 smoke run promote the whole
candidate panel on holdout. The script now defaults to `1,4,8`; use
`ENSEMBLE_KS=1,4,8,16` explicitly when full-panel P16 holdout is intended.
