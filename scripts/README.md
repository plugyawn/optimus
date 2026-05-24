# Optimus Scripts

The top-level scripts are supported launchers for the current Optimus workflow:

| script | purpose |
| --- | --- |
| `run_optimus_gpu_suite.sh` | Launch the P1024/P4096 search, halving, throughput, and report workflow. |
| `run_backend_parity_gate.sh` | Compare trusted HF/PEFT outputs against vLLM adapter-swapping outputs. |
| `prime_sync_and_run.sh` | Sync the checkout to a Prime pod and run smoke or GPU-suite workloads. |

Historical experiment recipes are not part of the supported repository surface.
Use the Optimus CLI and the documented runbooks instead.
