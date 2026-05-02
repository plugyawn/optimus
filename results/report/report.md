# RandOpt LoRA Lab Report

| run             | kind   | family    |   population |   base_screen_exact |   best_holdout_exact |   candidate_sec |   pair_sec | prompt_eval_savings   | best_tokens_per_sec   | best_batch_size   |
|:----------------|:-------|:----------|-------------:|--------------------:|---------------------:|----------------:|-----------:|:----------------------|:----------------------|:------------------|
| search_anzo_p32 | search | anzo      |           32 |              0.0625 |                0.125 |        0.452358 |    14.4754 |                       |                       |                   |
| search_anzo_p64 | search | anzo      |           64 |              0.0625 |                0.125 |        0.498147 |    15.9407 |                       |                       |                   |
| search_iso_p32  | search | isotropic |           32 |              0.0625 |                0     |        0.473247 |    15.1439 |                       |                       |                   |
| oracle          | oracle |           |            0 |              0.125  |              nan     |      nan        |   nan      |                       |                       |                   |
| search_iso_p64  | search | isotropic |           64 |              0.0625 |                0     |        0.499594 |    15.987  |                       |                       |                   |
