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

q + k + v + o matched relative norm:
  family=activation_spectral_lora_tscale_q2_k1p045_v1p045_o2
  targets=q_proj,k_proj,v_proj,o_proj
```

Use the same PEFT-confirmed quality gate as the two P64 panels. The selector is
not trusted, so report PEFT confirmation as authority.
