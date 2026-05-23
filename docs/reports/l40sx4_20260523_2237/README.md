# Optimus 4x L40S P1024/P4096 Report

This directory contains the committed public artifact bundle for the Prime 4x
L40S run completed on 2026-05-23.

Source run:

```text
results/prime_runs/l40sx4_20260523_2237/results
```

Run shape:

- hardware: Prime Crusoe 4x L40S 48GB;
- model: `Qwen/Qwen2.5-3B-Instruct`;
- runtime: vLLM 0.9.2, Torch 2.7.0 CUDA 12.6, Transformers 4.51.3;
- tensor parallelism: `TENSOR_PARALLEL_SIZE=4`;
- populations: `1024 4096`;
- screen prompts: `64`;
- holdout prompts: `256`;
- promoted candidates: `64`;
- adapter bench: `BENCH_ADAPTERS=8`;
- staged search: skipped with `RUN_HALVING=0`.

Key results:

| run | base holdout | screen-selected holdout | screen-selected delta | promoted holdout oracle | oracle delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| P1024 | `18/256` | `26/256` | `+8/256` | `38/256` | `+20/256` |
| P4096 | `18/256` | `28/256` | `+10/256` | `38/256` | `+20/256` |

Primary files:

- `report.md`: rendered systems and quality report.
- `validation.json`: strict run validation output.
- `quality_scaling.csv`: selector-vs-oracle quality table.
- `full_search.csv`: full-search throughput table.
- `bench.csv`: adapter benchmark throughput table.
- `best_of_n.csv`: best-of-N curve inputs.
- `*.png`: generated report plots.
