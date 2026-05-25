# Optimus Release Checklist

This checklist defines the minimum state before presenting Optimus as a public
research library.

Run the machine-readable gate before release:

```bash
optimus release-check \
  --gpu-root results/optimus_gpu_suite \
  --systems-out results/report/optimus_systems \
  --populations 1024,4096 \
  --strict
```

The public GitHub remote is the `optimus` repository, not the old
experiment-lab repository.

## Package Identity

- The installed package name is `optimus`.
- The public CLI is `optimus`.
- Public docs and examples use the `optimus` package and CLI only.
- The checkout has no top-level old experiment source namespace.
- Historical run outputs are not tracked under `results/`, and report bundles
  are not committed as public docs. Keep large or run-specific artifacts local
  under ignored result paths.

## Evidence Gates

- `optimus validate-run` passes for the referenced GPU suite.
- `optimus systems-report` separates screen-selected heldout transfer from
  promoted holdout-oracle quality.
- Subspace P1024 and P4096 reports include `subspace_state.pt`,
  `candidate_scores.jsonl`, `top_k_ensemble.json`, sample-level scorer details,
  token throughput, candidate/sec, and top-K/best-of-N data.
- Legacy LoRA baseline reports may include adapter rows and
  `candidate_summary.jsonl`, but those artifacts are not accepted as subspace
  evidence.
- Any selector-quality claim uses screen-selected heldout metrics.
- Any candidate-generation claim using holdout-oracle metrics is labeled as
  post-hoc promoted-candidate evidence.
- Fast-backend selector claims require a passing parity row in
  `parity.csv`; a header-only parity file is not release evidence.

## Systems Gates

- Prime or other rented GPU allocations are recorded in the ledger before
  launch.
- No rented GPU pod is active after a completed or failed run unless an
  immediate follow-up command is running.
- The worktree is clean and the checked commit is pushed to its upstream.

## Upstream

- Target the GitHub repository presented as `optimus`.
- Push the final branch to the Optimus repository only after the package/docs
  identity checks pass.
- Do not publish the old experiment-lab identity as the community-facing repo.
