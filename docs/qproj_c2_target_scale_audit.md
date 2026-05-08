# QProj C2 Target Scale Audit

## Run

```text
date: 2026-05-08
out: results/qproj_c2_target_scale_audit
model shape: Qwen2.5-3B style attention
hidden_size: 2048
num_attention_heads: 16
num_key_value_heads: 2
rank: 32
reference: q_proj at c=2
```

## Finding

The flat activation-spectral LoRA scale rule is not shape-fair across attention
targets. The materialized update has approximately:

```text
LoRA Frobenius / dense Gaussian Frobenius
  = c * (sqrt(out_features) + sqrt(in_features)) * sqrt(rank)
    / sqrt(out_features * in_features)
```

For rank 32 Qwen2.5-3B attention projections:

| target | shape | ratio at c=2 | c matching q c=2 ratio | c matching dense Frobenius |
| --- | ---: | ---: | ---: | ---: |
| q_proj | 2048x2048 | 0.5000 | 2.0000 | 4.0000 |
| k_proj | 256x2048 | 0.9571 | 1.0448 | 2.0896 |
| v_proj | 256x2048 | 0.9571 | 1.0448 | 2.0896 |
| o_proj | 2048x2048 | 0.5000 | 2.0000 | 4.0000 |

So a same-c q/v sweep was not a fair comparison. At c=2, k/v received nearly
twice the q/o relative dense norm.

## Code Support

Added target-specific activation-spectral family syntax:

```text
activation_spectral_lora_tscale_q2_v1
activation_spectral_lora_tscale_q2_v1p045
activation_spectral_lora_tscale_q2_k1_v1_o2
activation_spectral_lora_tscale_q2_k1p045_v1p045_o2
```

The matched-reference q/k/v/o family emitted by the audit is:

```text
activation_spectral_lora_tscale_q2_k1p045_v1p045_o2
```

This equalizes each target's LoRA/dense Frobenius ratio, but it does not keep
the total update budget equal to q-only. The matched-reference q+v arm has
`1.061x` the q-only total Frobenius; the q+k+v+o arm has `1.5x`.

The audit therefore also emits global-budget-matched families:

```text
activation_spectral_lora_tscale_q1p886_v0p985
activation_spectral_lora_tscale_q1p333_k0p697_v0p697_o1p333
```

## Next Test

The next GPU experiment should compare q-only c2 against shape-normalized mixed
targets, not same-c mixed targets:

```text
q only:
  family=activation_spectral_lora_c2
  targets=q_proj

q + v matched relative norm:
  family=activation_spectral_lora_tscale_q2_v1p045
  targets=q_proj,v_proj

q + v matched total norm:
  family=activation_spectral_lora_tscale_q1p886_v0p985
  targets=q_proj,v_proj

q + k + v + o matched relative norm:
  family=activation_spectral_lora_tscale_q2_k1p045_v1p045_o2
  targets=q_proj,k_proj,v_proj,o_proj

q + k + v + o matched total norm:
  family=activation_spectral_lora_tscale_q1p333_k0p697_v0p697_o1p333
  targets=q_proj,k_proj,v_proj,o_proj
```

Use the same PEFT-confirmed quality gate as the two P64 panels. The selector is
not trusted, so report PEFT confirmation as authority.

## P64 Outcome

The first target-scaled mixed-target GPU check is recorded in
`docs/target_scaled_mixed_targets_p64_l40s.md`.

```text
q+v matched relative norm: 9/128 strict holdout
q+v matched total norm:   10/128 strict holdout
q+k+v+o matched total:    11/128 strict holdout
strong-positive gate:     13/128 strict holdout
```

All runs passed validity and the vLLM path stayed 5x-8x faster after PEFT
confirmation, but no arm cleared the quality gate. Do not scale this target
scaling branch without a stronger selector or parameterization signal.
