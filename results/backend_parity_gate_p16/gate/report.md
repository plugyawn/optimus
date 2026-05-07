# Backend Parity Gate

Status: **FAIL**

| gate | pass |
| --- | ---: |
| protocol metadata | True |
| base rows present | True |
| ranking correlation | False |
| adapter tensor parity | True |

## Ranking

| metric | value |
| --- | ---: |
| common candidates | 16 |
| Spearman | -0.1811643254631353 |
| top8 overlap | 7/8 |
| selected regret vs trusted | 0.125 |

## Checks

| check | pass | trusted | candidate | note |
| --- | ---: | --- | --- | --- |
| summary.family | True | factor_gaussian_lora | factor_gaussian_lora |  |
| summary.population | True | 16 | 16 |  |
| summary.rank | True | 8 | 8 |  |
| summary.sigma | True | 0.0075 | 0.0075 |  |
| summary.targets | True | q_proj,v_proj | q_proj,v_proj |  |
| summary.screen_prompts | True | 16 | 16 |  |
| summary.max_new_tokens | True | 32 | 32 |  |
| summary.stop_at_answer | True | True | True |  |
| summary.antithetic | True | True | True |  |
| trusted.screen_holdout_overlap_zero | True | 0 |  |  |
| candidate.screen_holdout_overlap_zero | True | 0 |  |  |
| peft.base_screen_rows_present | True | results/backend_parity_gate_p16/peft/per_prompt.jsonl |  |  |
| peft.base_holdout_rows_present | True | results/backend_parity_gate_p16/peft/holdout_per_prompt.jsonl |  |  |
| vllm.base_screen_rows_present | True | results/backend_parity_gate_p16/vllm/per_prompt.jsonl |  |  |
| vllm.base_holdout_rows_present | True | results/backend_parity_gate_p16/vllm/holdout_per_prompt.jsonl |  |  |
| model.layers.0.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.0.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.0.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.0.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.1.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.1.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.1.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.1.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.2.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.2.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.2.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.2.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.3.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.3.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.3.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.3.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.4.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.4.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.4.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.4.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.5.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.5.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.5.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.5.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.6.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.6.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.6.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.6.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.7.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.7.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.7.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.7.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.8.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.8.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.8.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.8.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.9.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.9.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.9.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.9.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.10.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.10.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.10.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.10.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.11.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.11.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.11.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.11.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.12.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.12.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.12.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.12.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.13.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.13.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.13.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.13.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.14.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.14.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.14.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.14.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.15.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.15.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.15.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.15.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.16.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.16.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.16.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.16.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.17.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.17.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.17.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.17.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.18.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.18.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.18.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.18.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.19.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.19.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.19.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.19.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.20.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.20.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.20.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.20.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.21.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.21.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.21.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.21.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.22.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.22.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.22.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.22.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.23.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.23.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.23.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.23.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.24.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.24.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.24.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.24.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.25.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.25.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.25.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.25.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.26.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.26.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.26.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.26.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.27.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.27.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.27.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.27.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.28.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.28.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.28.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.28.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.29.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.29.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.29.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.29.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.30.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.30.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.30.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.30.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.31.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.31.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.31.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.31.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.32.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.32.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.32.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.32.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.33.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.33.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.33.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.33.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.34.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.34.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.34.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.34.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.35.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.35.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.35.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.35.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_pos | factor_gaussian_lora:seed1019282515:s0.0075:sign1 |  |
| model.layers.0.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.0.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.0.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.0.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.1.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.1.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.1.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.1.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.2.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.2.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.2.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.2.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.3.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.3.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.3.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.3.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.4.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.4.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.4.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.4.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.5.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.5.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.5.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.5.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.6.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.6.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.6.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.6.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.7.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.7.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.7.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.7.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.8.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.8.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.8.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.8.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.9.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.9.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.9.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.9.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.10.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.10.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.10.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.10.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.11.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.11.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.11.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.11.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.12.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.12.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.12.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.12.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.13.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.13.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.13.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.13.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.14.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.14.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.14.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.14.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.15.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.15.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.15.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.15.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.16.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.16.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.16.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.16.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.17.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.17.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.17.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.17.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.18.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.18.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.18.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.18.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.19.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.19.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.19.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.19.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.20.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.20.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.20.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.20.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.21.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.21.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.21.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.21.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.22.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.22.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.22.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.22.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.23.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.23.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.23.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.23.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.24.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.24.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.24.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.24.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.25.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.25.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.25.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.25.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.26.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.26.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.26.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.26.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.27.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.27.self_attn.q_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.27.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
| model.layers.27.self_attn.v_proj | True | randopt_seed1019282515_s0.0075_neg | factor_gaussian_lora:seed1019282515:s0.0075:sign-1 |  |
