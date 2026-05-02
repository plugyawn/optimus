# RandOpt LoRA Lab

Five-hour experiment lab for accelerating perturbation search on one A100/H100-class GPU.

The lab tests two axes:

- Systems: how fast can we evaluate many LoRA perturbation candidates without corrupting the base model?
- Geometry: can structured candidate distributions make useful perturbations appear more often?

Default task is Countdown with an answer-only expression prompt and exact expression reward.

## Run Targets

```bash
python -m randopt_lora_lab.experiments oracle --out results/oracle
python -m randopt_lora_lab.experiments search --out results/search_iso --family isotropic --population 64
python -m randopt_lora_lab.experiments search --out results/search_anzo --family anzo --population 64
python -m randopt_lora_lab.report --root results --out results/report
```

The first run is a correctness oracle. It must pass before any throughput or research claim.

## Evidence Rules

- Zero LoRA must match base outputs.
- Base outputs must match after candidate cycles.
- Log per-prompt rewards, not just aggregate means.
- Report cap-hit, malformed, and exact-answer rates separately.
- Compare lift vs base on the same prompt split.
