# Shortlist Alignment Audit

Run root: `results/tscale_qv_p64_seed20260507`

## Counts

| item | count |
| --- | ---: |
| dense | 64 |
| proposal | 64 |
| confirmed | 8 |
| proposal_conditions | 192 |

## Dense vs vLLM Proposal

| metric | common | spearman | pearson | mean abs delta | max abs delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| selection_score | 64 | -0.134335 | -0.286419 | 0.112915 | 0.664062 |
| proposal_exact | 64 | -0.0329868 | -0.0760956 | 0.0262451 | 0.0625 |
| default_exact | 64 | 0.00400608 | 0.0267852 | 0.0229492 | 0.078125 |
| reordered_exact | 64 | -0.00998404 | -0.111042 | 0.0407715 | 0.078125 |

### Dense Recall By vLLM Selection

Dense best spec: `seed239990546:s0.002:sign1`
Dense best proposal rank: `14`

| k | contains dense best | dense top-k overlap |
| ---: | --- | ---: |
| 1 | false | 0 |
| 2 | false | 0 |
| 4 | false | 0 |
| 8 | false | 0 |
| 16 | true | 4 |
| 32 | true | 15 |

## Confirmed PEFT vs vLLM Proposal

| metric | common | spearman | pearson | mean abs delta | max abs delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| selection_score | 8 | -0.00680288 | -0.0365284 | 0.0400391 | 0.078125 |
| proposal_exact | 8 | -0.321918 | -0.239437 | 0.0341797 | 0.0625 |
| default_exact | 8 | 0.289699 | 0.318091 | 0.0195312 | 0.0625 |
| reordered_exact | 8 | -0.672987 | -0.751439 | 0.0488281 | 0.078125 |

## vLLM Default vs Reordered

| metric | common | spearman | pearson | mean abs delta | max abs delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| exact | 64 | -0.0524948 | -0.0465316 | 0.0368652 | 0.125 |
| selection | 64 | 0.060782 | 0.0103435 | 0.165283 | 1.1875 |

## Per-Prompt Default Backend Agreement

| metric | value |
| --- | ---: |
| common rows | 512 |
| exact equal fraction | 0.964844 |
| text equal fraction | 0.552734 |
| vLLM exact mean | 0.0800781 |
| PEFT exact mean | 0.0605469 |

## Interpretation

This audit separates the shortlist failure into three checks:

1. Whether vLLM proposal scores agree with dense scores across the full population.
2. Whether vLLM proposal scores agree with trusted PEFT scores on the confirmed shortlist.
3. Whether default-prompt vLLM and PEFT outputs are identical enough to trust vLLM as a selector.
