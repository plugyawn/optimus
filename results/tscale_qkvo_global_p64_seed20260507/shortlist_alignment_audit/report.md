# Shortlist Alignment Audit

Run root: `results/tscale_qkvo_global_p64_seed20260507`

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
| selection_score | 64 | -0.310136 | -0.138864 | 0.123169 | 0.632812 |
| proposal_exact | 64 | -0.107917 | -0.103046 | 0.0268555 | 0.078125 |
| default_exact | 64 | 0.00327639 | 0.0129629 | 0.020752 | 0.078125 |
| reordered_exact | 64 | -0.0855284 | -0.142056 | 0.0412598 | 0.09375 |

### Dense Recall By vLLM Selection

Dense best spec: `seed239990546:s0.002:sign1`
Dense best proposal rank: `49`

| k | contains dense best | dense top-k overlap |
| ---: | --- | ---: |
| 1 | false | 0 |
| 2 | false | 0 |
| 4 | false | 0 |
| 8 | false | 0 |
| 16 | false | 2 |
| 32 | false | 13 |

## Confirmed PEFT vs vLLM Proposal

| metric | common | spearman | pearson | mean abs delta | max abs delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| selection_score | 8 | 0.601938 | 0.506024 | 0.0458984 | 0.09375 |
| proposal_exact | 8 | 0.601938 | 0.362103 | 0.0322266 | 0.0625 |
| default_exact | 8 | -0.333443 | -0.199108 | 0.0332031 | 0.0625 |
| reordered_exact | 8 | 0.797532 | 0.819892 | 0.0429688 | 0.078125 |

## vLLM Default vs Reordered

| metric | common | spearman | pearson | mean abs delta | max abs delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| exact | 64 | 0.119017 | 0.224312 | 0.0332031 | 0.078125 |
| selection | 64 | 0.0776917 | 0.153648 | 0.182861 | 1.17188 |

## Per-Prompt Default Backend Agreement

| metric | value |
| --- | ---: |
| common rows | 512 |
| exact equal fraction | 0.955078 |
| text equal fraction | 0.548828 |
| vLLM exact mean | 0.0800781 |
| PEFT exact mean | 0.0703125 |

## Interpretation

This audit separates the shortlist failure into three checks:

1. Whether vLLM proposal scores agree with dense scores across the full population.
2. Whether vLLM proposal scores agree with trusted PEFT scores on the confirmed shortlist.
3. Whether default-prompt vLLM and PEFT outputs are identical enough to trust vLLM as a selector.
