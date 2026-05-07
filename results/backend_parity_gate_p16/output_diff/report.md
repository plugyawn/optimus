# Backend Output Diff

| metric | value |
| --- | ---: |
| common rows | 256 |
| common candidates | 16 |
| exact disagreement rate | 0.0234375 |
| answer equal rate | 0.67578125 |
| text equal rate | 0.55078125 |
| max abs exact delta by candidate | 0.125 |

## Worst Candidate Deltas

| candidate | n | peft_exact_mean | vllm_exact_mean | exact_delta | exact_disagreement_rate | answer_equal_rate | text_equal_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| factor_gaussian_lora:seed509771609:s0.0075:sign-1 | 16 | 0.125 | 0.0 | -0.125 | 0.125 | 0.5625 | 0.5625 |
| factor_gaussian_lora:seed1019282515:s0.0075:sign1 | 16 | 0.0 | 0.0625 | 0.0625 | 0.0625 | 0.4375 | 0 |
| factor_gaussian_lora:seed1530295774:s0.0075:sign-1 | 16 | 0.0 | 0.0625 | 0.0625 | 0.0625 | 0.875 | 0.875 |
| factor_gaussian_lora:seed1837932928:s0.0075:sign-1 | 16 | 0.0 | 0.0625 | 0.0625 | 0.0625 | 0.25 | 0.25 |
| factor_gaussian_lora:seed1837932928:s0.0075:sign1 | 16 | 0.0625 | 0.0 | -0.0625 | 0.0625 | 0.4375 | 0.4375 |
| factor_gaussian_lora:seed1019282515:s0.0075:sign-1 | 16 | 0.0 | 0.0 | 0.0 | 0 | 0.5625 | 0.5 |
| factor_gaussian_lora:seed1442006728:s0.0075:sign-1 | 16 | 0.0 | 0.0 | 0.0 | 0 | 0.875 | 0.8125 |
| factor_gaussian_lora:seed1442006728:s0.0075:sign1 | 16 | 0.0 | 0.0 | 0.0 | 0 | 0.625 | 0.4375 |
| factor_gaussian_lora:seed1530295774:s0.0075:sign1 | 16 | 0.0 | 0.0 | 0.0 | 0 | 0.6875 | 0.6875 |
| factor_gaussian_lora:seed1679872250:s0.0075:sign-1 | 16 | 0.0 | 0.0 | 0.0 | 0 | 0.9375 | 0.9375 |
| factor_gaussian_lora:seed1679872250:s0.0075:sign1 | 16 | 0.0 | 0.0 | 0.0 | 0 | 0.875 | 0.0625 |
| factor_gaussian_lora:seed221822464:s0.0075:sign-1 | 16 | 0.0 | 0.0 | 0.0 | 0 | 0.75 | 0.6875 |
| factor_gaussian_lora:seed221822464:s0.0075:sign1 | 16 | 0.0 | 0.0 | 0.0 | 0 | 0.9375 | 0.9375 |
| factor_gaussian_lora:seed385390264:s0.0075:sign-1 | 16 | 0.0 | 0.0 | 0.0 | 0 | 0.3125 | 0.3125 |
| factor_gaussian_lora:seed385390264:s0.0075:sign1 | 16 | 0.0 | 0.0 | 0.0 | 0 | 1 | 0.625 |
| factor_gaussian_lora:seed509771609:s0.0075:sign1 | 16 | 0.0 | 0.0 | 0.0 | 0 | 0.6875 | 0.6875 |
