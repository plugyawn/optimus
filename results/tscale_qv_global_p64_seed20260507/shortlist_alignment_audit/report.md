# Shortlist Alignment Audit

Run root: `results/tscale_qv_global_p64_seed20260507`

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
| selection_score | 64 | -0.143825 | -0.279713 | 0.108154 | 0.65625 |
| proposal_exact | 64 | -0.0515385 | -0.0894169 | 0.0269775 | 0.0625 |
| default_exact | 64 | -0.0415185 | 0 | 0.0241699 | 0.078125 |
| reordered_exact | 64 | 0.0130134 | -0.113401 | 0.0405273 | 0.078125 |

### Dense Recall By vLLM Selection

Dense best spec: `seed239990546:s0.002:sign1`
Dense best proposal rank: `28`

| k | contains dense best | dense top-k overlap |
| ---: | --- | ---: |
| 1 | false | 0 |
| 2 | false | 0 |
| 4 | false | 0 |
| 8 | false | 0 |
| 16 | false | 3 |
| 32 | true | 15 |

## Confirmed PEFT vs vLLM Proposal

| metric | common | spearman | pearson | mean abs delta | max abs delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| selection_score | 8 | 0.1218 | 0.145248 | 0.0351562 | 0.0625 |
| proposal_exact | 8 | 0.1218 | 0.145248 | 0.0351562 | 0.0625 |
| default_exact | 8 | 0.159028 | 0.150946 | 0.0292969 | 0.0625 |
| reordered_exact | 8 | -0.175933 | -0.0425243 | 0.0410156 | 0.078125 |

## vLLM Default vs Reordered

| metric | common | spearman | pearson | mean abs delta | max abs delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| exact | 64 | -0.0340969 | -0.00963037 | 0.0349121 | 0.125 |
| selection | 64 | 0.0340352 | 0.0497101 | 0.158203 | 1.1875 |

## Per-Prompt Default Backend Agreement

| metric | value |
| --- | ---: |
| common rows | 512 |
| exact equal fraction | 0.958984 |
| text equal fraction | 0.552734 |
| vLLM exact mean | 0.0859375 |
| PEFT exact mean | 0.0566406 |

## Interpretation

This audit separates the shortlist failure into three checks:

1. Whether vLLM proposal scores agree with dense scores across the full population.
2. Whether vLLM proposal scores agree with trusted PEFT scores on the confirmed shortlist.
3. Whether default-prompt vLLM and PEFT outputs are identical enough to trust vLLM as a selector.
