# Backend Output Diff

| metric | value |
| --- | ---: |
| common rows | 4096 |
| common candidates | 64 |
| exact disagreement rate | 0.02392578125 |
| answer equal rate | 0.601806640625 |
| text equal rate | 0.529052734375 |
| max abs exact delta by candidate | 0.125 |

## Worst Candidate Deltas

| candidate | n | peft_exact_mean | vllm_token_ids_exact_mean | exact_delta | exact_disagreement_rate | answer_equal_rate | text_equal_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| factor_gaussian_lora:seed1632697641:s0.0075:sign-1 | 64 | 0.0 | 0.125 | 0.125 | 0.125 | 0.3125 | 0.265625 |
| factor_gaussian_lora:seed1411240924:s0.0075:sign1 | 64 | 0.21875 | 0.171875 | -0.046875 | 0.046875 | 0.6875 | 0.6875 |
| factor_gaussian_lora:seed1825954996:s0.0075:sign1 | 64 | 0.0 | 0.046875 | 0.046875 | 0.046875 | 0.3125 | 0.234375 |
| factor_gaussian_lora:seed1837932928:s0.0075:sign-1 | 64 | 0.015625 | 0.0625 | 0.046875 | 0.046875 | 0.328125 | 0.328125 |
| factor_gaussian_lora:seed326653716:s0.0075:sign1 | 64 | 0.0 | 0.046875 | 0.046875 | 0.046875 | 0.3125 | 0.125 |
| factor_gaussian_lora:seed513262593:s0.0075:sign1 | 64 | 0.0 | 0.046875 | 0.046875 | 0.046875 | 0.46875 | 0.421875 |
| factor_gaussian_lora:seed92382467:s0.0075:sign-1 | 64 | 0.015625 | 0.0625 | 0.046875 | 0.046875 | 0.546875 | 0.515625 |
| factor_gaussian_lora:seed1019282515:s0.0075:sign1 | 64 | 0.03125 | 0.0625 | 0.03125 | 0.0625 | 0.453125 | 0.125 |
| factor_gaussian_lora:seed1156649635:s0.0075:sign-1 | 64 | 0.0625 | 0.09375 | 0.03125 | 0.0625 | 0.75 | 0.703125 |
| factor_gaussian_lora:seed221822464:s0.0075:sign-1 | 64 | 0.109375 | 0.140625 | 0.03125 | 0.0625 | 0.65625 | 0.53125 |
| factor_gaussian_lora:seed326653716:s0.0075:sign-1 | 64 | 0.046875 | 0.078125 | 0.03125 | 0.0625 | 0.578125 | 0.578125 |
| factor_gaussian_lora:seed513262593:s0.0075:sign-1 | 64 | 0.03125 | 0.0625 | 0.03125 | 0.0625 | 0.53125 | 0.5 |
| factor_gaussian_lora:seed1167280424:s0.0075:sign-1 | 64 | 0.0 | 0.03125 | 0.03125 | 0.03125 | 0.546875 | 0.125 |
| factor_gaussian_lora:seed1407710298:s0.0075:sign-1 | 64 | 0.0625 | 0.03125 | -0.03125 | 0.03125 | 0.65625 | 0.65625 |
| factor_gaussian_lora:seed1825954996:s0.0075:sign-1 | 64 | 0.0 | 0.03125 | 0.03125 | 0.03125 | 0.375 | 0.25 |
| factor_gaussian_lora:seed1837932928:s0.0075:sign1 | 64 | 0.078125 | 0.046875 | -0.03125 | 0.03125 | 0.5625 | 0.546875 |
