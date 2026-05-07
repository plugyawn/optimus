# Backend Output Diff

| metric | value |
| --- | ---: |
| common rows | 4096 |
| common candidates | 64 |
| exact disagreement rate | 0.02587890625 |
| answer equal rate | 0.597900390625 |
| text equal rate | 0.526611328125 |
| max abs exact delta by candidate | 0.125 |

## Worst Candidate Deltas

| candidate | n | peft_exact_mean | vllm_exact_mean | exact_delta | exact_disagreement_rate | answer_equal_rate | text_equal_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| factor_gaussian_lora:seed1632697641:s0.0075:sign-1 | 64 | 0.0 | 0.125 | 0.125 | 0.125 | 0.3125 | 0.25 |
| factor_gaussian_lora:seed1019282515:s0.0075:sign1 | 64 | 0.03125 | 0.09375 | 0.0625 | 0.09375 | 0.421875 | 0.125 |
| factor_gaussian_lora:seed1825954996:s0.0075:sign1 | 64 | 0.0 | 0.0625 | 0.0625 | 0.0625 | 0.28125 | 0.203125 |
| factor_gaussian_lora:seed326653716:s0.0075:sign1 | 64 | 0.0 | 0.0625 | 0.0625 | 0.0625 | 0.296875 | 0.140625 |
| factor_gaussian_lora:seed221822464:s0.0075:sign-1 | 64 | 0.109375 | 0.15625 | 0.046875 | 0.078125 | 0.625 | 0.484375 |
| factor_gaussian_lora:seed1837932928:s0.0075:sign-1 | 64 | 0.015625 | 0.0625 | 0.046875 | 0.046875 | 0.28125 | 0.28125 |
| factor_gaussian_lora:seed513262593:s0.0075:sign1 | 64 | 0.0 | 0.046875 | 0.046875 | 0.046875 | 0.484375 | 0.453125 |
| factor_gaussian_lora:seed92382467:s0.0075:sign-1 | 64 | 0.015625 | 0.0625 | 0.046875 | 0.046875 | 0.53125 | 0.5 |
| factor_gaussian_lora:seed1099262776:s0.0075:sign1 | 64 | 0.09375 | 0.125 | 0.03125 | 0.0625 | 0.71875 | 0.71875 |
| factor_gaussian_lora:seed1411240924:s0.0075:sign1 | 64 | 0.21875 | 0.1875 | -0.03125 | 0.0625 | 0.640625 | 0.625 |
| factor_gaussian_lora:seed326653716:s0.0075:sign-1 | 64 | 0.046875 | 0.078125 | 0.03125 | 0.0625 | 0.5625 | 0.5625 |
| factor_gaussian_lora:seed513262593:s0.0075:sign-1 | 64 | 0.03125 | 0.0625 | 0.03125 | 0.0625 | 0.5625 | 0.5625 |
| factor_gaussian_lora:seed1167280424:s0.0075:sign-1 | 64 | 0.0 | 0.03125 | 0.03125 | 0.03125 | 0.5625 | 0.15625 |
| factor_gaussian_lora:seed1167280424:s0.0075:sign1 | 64 | 0.015625 | 0.046875 | 0.03125 | 0.03125 | 0.703125 | 0.703125 |
| factor_gaussian_lora:seed1442006728:s0.0075:sign1 | 64 | 0.0 | 0.03125 | 0.03125 | 0.03125 | 0.5 | 0.21875 |
| factor_gaussian_lora:seed1825954996:s0.0075:sign-1 | 64 | 0.0 | 0.03125 | 0.03125 | 0.03125 | 0.375 | 0.265625 |
