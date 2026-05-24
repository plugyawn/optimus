# Optimus Release Checklist

This checklist defines the minimum state before presenting Optimus as a public
research library.

Run the machine-readable gate before release:

```bash
optimus release-check \
  --gpu-root results/prime_runs/l40sx4_20260523_2237/results/optimus_gpu_suite_v092_noflash_tp4 \
  --systems-out results/prime_runs/l40sx4_20260523_2237/results/report/optimus_systems_v092_noflash_tp4 \
  --populations 1024,4096 \
  --bench-adapters 8 \
  --skip-halving \
  --strict
```

This gate passes for the current 4x L40S release evidence when the fetched GPU
artifacts are present locally. The public GitHub remote is the `optimus`
repository, not the old experiment-lab repository.

## Package Identity

- The installed package name is `optimus`.
- The public CLI is `optimus`.
- Public docs and examples use the `optimus` package and CLI only.
- The checkout has no top-level old experiment source namespace.
- Historical run outputs are not tracked under `results/`; curated report
  artifacts live under `docs/reports/`.

## Evidence Gates

- `optimus validate-run` passes for the referenced GPU suite.
- `optimus systems-report` separates screen-selected heldout transfer from
  promoted holdout-oracle quality.
- P1024 and P4096 reports include per-prompt rows, candidate summaries, token
  throughput, candidate/sec, and best-of-N data.
- Any selector-quality claim uses screen-selected heldout metrics.
- Any candidate-generation claim using holdout-oracle metrics is labeled as
  post-hoc promoted-candidate evidence.

## Systems Gates

- Prime or other rented GPU allocations are recorded in the ledger before
  launch.
- No rented GPU pod is active after a completed or failed run unless an
  immediate follow-up command is running.
- The intended 8xA100-class run is either completed and linked, or explicitly
  listed as optional larger-systems evidence beyond the accepted 4x fallback.

## Upstream

- Target the GitHub repository presented as `optimus`.
- Push the final branch to the Optimus repository only after the package/docs
  identity checks pass.
- Do not publish the old experiment-lab identity as the community-facing repo.
