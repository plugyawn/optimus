# Shortlist Alignment Audit

Run root: `results/qproj_c2_vllm_shortlist_p64_seed20260508`

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
| selection_score | 64 | 0.082039 | -0.0304546 | 0.083374 | 0.398438 |
| proposal_exact | 64 | 0.0715169 | 0.0317288 | 0.0167236 | 0.0625 |
| default_exact | 64 | 0.171311 | 0.153291 | 0.0168457 | 0.046875 |
| reordered_exact | 64 | -0.038131 | -0.106289 | 0.0224609 | 0.078125 |

### Dense Recall By vLLM Selection

Dense best spec: `seed931683830:s0.001:sign1`
Dense best proposal rank: `31`

| k | contains dense best | dense top-k overlap |
| ---: | --- | ---: |
| 1 | false | 0 |
| 2 | false | 0 |
| 4 | false | 0 |
| 8 | false | 1 |
| 16 | false | 5 |
| 32 | true | 17 |

## Confirmed PEFT vs vLLM Proposal

| metric | common | spearman | pearson | mean abs delta | max abs delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| selection_score | 8 | 0.233882 | 0.22771 | 0.0263672 | 0.0546875 |
| proposal_exact | 8 | 0.233882 | 0.22771 | 0.0283203 | 0.046875 |
| default_exact | 8 | 0.0897436 | 0.111111 | 0.03125 | 0.0625 |
| reordered_exact | 8 | 0 | 0 | 0.0292969 | 0.0625 |

## vLLM Default vs Reordered

| metric | common | spearman | pearson | mean abs delta | max abs delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| exact | 64 | 0.160567 | 0.166943 | 0.0212402 | 0.0625 |
| selection | 64 | 0.139166 | 0.162291 | 0.0671387 | 0.609375 |

## Per-Prompt Default Backend Agreement

| metric | value |
| --- | ---: |
| common rows | 512 |
| exact equal fraction | 0.953125 |
| text equal fraction | 0.605469 |
| vLLM exact mean | 0.0839844 |
| PEFT exact mean | 0.0527344 |

## Interpretation

This audit separates the shortlist failure into three checks:

1. Whether vLLM proposal scores agree with dense scores across the full population.
2. Whether vLLM proposal scores agree with trusted PEFT scores on the confirmed shortlist.
3. Whether default-prompt vLLM and PEFT outputs are identical enough to trust vLLM as a selector.
