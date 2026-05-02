# RandOpt LoRA Lab Report

| run                  | kind            | family    |   population |   base_screen_exact |   best_holdout_exact |   candidate_sec |   pair_sec |   prompt_eval_savings |   best_tokens_per_sec |   best_batch_size |
|:---------------------|:----------------|:----------|-------------:|--------------------:|---------------------:|----------------:|-----------:|----------------------:|----------------------:|------------------:|
| search_anzo_p32      | search          | anzo      |           32 |              0.0625 |              0.125   |        0.452358 |    14.4754 |            nan        |                nan    |               nan |
| vllm_base_probe      | vllm_base_probe |           |            0 |            nan      |            nan       |      nan        |   nan      |            nan        |                nan    |               nan |
| sigma_iso_p64_s0p01  | search          | isotropic |           64 |              0.0625 |              0.15625 |        0.49918  |    15.9738 |            nan        |                nan    |               nan |
| search_anzo_p64      | search          | anzo      |           64 |              0.0625 |              0.125   |        0.498147 |    15.9407 |            nan        |                nan    |               nan |
| search_iso_p32       | search          | isotropic |           32 |              0.0625 |              0       |        0.473247 |    15.1439 |            nan        |                nan    |               nan |
| sigma_iso_p64_s0p04  | search          | isotropic |           64 |              0.0625 |              0       |        0.499252 |    15.9761 |            nan        |                nan    |               nan |
| halving_iso_p128     | halving         | isotropic |          128 |              0.0625 |              0       |        0.816712 |   nan      |              0.606061 |                nan    |               nan |
| sigma_iso_p64_s0p005 | search          | isotropic |           64 |              0.0625 |              0.125   |        0.50542  |    16.1735 |            nan        |                nan    |               nan |
| halving_anzo_p128    | halving         | anzo      |          128 |              0.0625 |              0.125   |        0.816943 |   nan      |              0.606061 |                nan    |               nan |
| sysbench_tf3b        | sysbench        | isotropic |            0 |            nan      |            nan       |      nan        |   nan      |            nan        |               1034.96 |                32 |
| oracle               | oracle          |           |            0 |              0.125  |            nan       |      nan        |   nan      |            nan        |                nan    |               nan |
| sigma_iso_p64_s0p02  | search          | isotropic |           64 |              0.0625 |              0       |        0.502273 |    16.0727 |            nan        |                nan    |               nan |
| search_iso_p64       | search          | isotropic |           64 |              0.0625 |              0       |        0.499594 |    15.987  |            nan        |                nan    |               nan |
