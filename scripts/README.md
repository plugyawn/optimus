# Optimus Scripts

The top-level scripts are supported launchers for the current Optimus workflow:

| script | purpose |
| --- | --- |
| `run_optimus_gpu_suite.sh` | Launch the P1024/P4096 search, halving, throughput, and report workflow. |
| `run_backend_parity_gate.sh` | Compare trusted HF/PEFT outputs against vLLM adapter-swapping outputs. |
| `prime_sync_and_run.sh` | Sync the checkout to a Prime pod and run smoke or GPU-suite workloads. |

Older experiment and maintenance recipes are archived under `archive/` for
provenance. They are not the supported Optimus interface.
