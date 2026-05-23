# Optimus Release Checklist

This checklist defines the minimum state before presenting Optimus as a public
research library.

Run the machine-readable gate before release:

```bash
optimus release-check \
  --gpu-root results/prime_runs/l40sx2_20260523_2134/results/optimus_gpu_suite_v092_noflash \
  --systems-out results/prime_runs/l40sx2_20260523_2134/results/report/optimus_systems_v092_noflash \
  --populations 1024,4096 \
  --bench-adapters 8 \
  --skip-halving \
  --strict
```

The gate is expected to fail until every listed release blocker is fixed. In
this checkout, the final GitHub remote must still be changed from the old
experiment-lab repository to an `optimus` repository before publishing.

## Package Identity

- The installed package name is `optimus`.
- The public CLI is `optimus`.
- Public docs and examples do not instruct users to import or run
  `randopt_lora_lab`.
- Historical experiment modules, if still present in the source checkout, are
  excluded from the published package and Optimus CLI.

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
  listed as a remaining gap.

## Upstream

- Create or target a GitHub repository presented as `optimus`, not
  `randopt-lora-lab`.
- Push the final branch to the Optimus repository only after the package/docs
  identity checks pass.
- Do not publish the old experiment-lab identity as the community-facing repo.
