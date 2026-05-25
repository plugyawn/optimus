# Optimus Scripts

The top-level scripts are supported launchers for the current Optimus workflow:

| script | purpose |
| --- | --- |
| `run_optimus_gpu_suite.sh` | Launch the explicit LoRA baseline or subspace P1024/P4096 search plan, throughput checks where implemented, and report workflow. Staged search is disabled until it has a final public route. |
| `run_population_lighteval_pipeline.sh` | Run search populations, materialize selected adapters, evaluate with LightEval, and generate plots. |
| `run_lighteval_population_sweep.sh` | Plan or run LightEval over population-labelled model artifacts. |
| `run_backend_parity_gate.sh` | Compare trusted HF/PEFT outputs against vLLM adapter-swapping outputs. |
| `prime_sync_and_run.sh` | Sync the checkout to a Prime pod and run smoke, GPU-suite, LightEval-sweep, or full population-LightEval workloads. |

Historical experiment recipes are not part of the supported repository surface.
Use the Optimus CLI and the documented runbooks instead.
