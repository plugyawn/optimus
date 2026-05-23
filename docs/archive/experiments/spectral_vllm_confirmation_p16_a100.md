# Spectral vLLM Confirmation P16 A100

## Run

```text
date: 2026-05-08
pod: randopt-spectral-vllm, Prime pod 6f9788b8dac5460d9d9e2163ea5b0248
gpu: 1x NVIDIA A100-SXM4-80GB
repo base commit: d83dcde6f6f23e7f054e6edcad704e3af5b963fa
out: results/spectral_vllm_confirmation_rank32_c1p5_p16_default
model: Qwen/Qwen2.5-3B-Instruct
population: 16
screen prompts: 64
holdout prompts: 128 for PEFT arms, 8 for vLLM proposal
max_new_tokens: 128
prompt variants: default only
sigma values: 0.0005, 0.001, 0.002
rank: 32
family: spectral_projected_gaussian_rank_r_c1p5
targets: q_proj, v_proj
```

The pod artifacts were copied locally before shutdown. The Prime pod was then
terminated, and `prime pods list -o json` returned an empty pod list.

## Result

This is a useful systems positive and a quality negative.

The vLLM proposal-plus-PEFT-confirmation path passed its economics gate:

```text
trusted full PEFT spectral screen: 1169.75 s
vLLM proposal screen: 3.63 s
vLLM load/build: 87.48 s
zero-regret k: 2
best recovered k: 4
eval-only speedup at k=2: 26.39x
full-without-PEFT-load speedup at k=2: 8.87x
```

But the dense-parity gate still failed:

| arm | screen best | best ensemble holdout | Spearman vs dense | top-8 overlap | selected regret | parity pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| dense Gaussian | 9.375% | 8.594% | n/a | n/a | n/a | reference |
| spectral rank-32 c1.5 | 9.375% | 8.594% | 0.300 | 7/8 | 1.5625 pp | false |
| factor LoRA control | 9.375% | 7.8125% | -0.009 | 4/8 | 3.125 pp | false |

The spectral family did tie dense on the best ensemble holdout metric in this
small panel, but it did not reproduce dense's candidate ranking or selected
best candidate. That means the current evidence supports vLLM as a fast
proposal engine for a matched adapter family, not spectral LoRA as a proven
replacement for dense Gaussian RandOpt.

## Validity

The PEFT validity gates passed for dense, control, and spectral. On the promoted
candidates, cap hits were essentially gone at `max_new_tokens=128`:

```text
dense selected cap-hit: 0.0
control selected cap-hit: 0.0
spectral selected cap-hit: 0.0
spectral selected malformed: 0.0
```

The vLLM proposal base check was also clean on the default prompt:

```text
base screen exact: 3.125%
base screen cap-hit: 0.0
base screen malformed: 0.0
screen throughput: 281.81 prompts/s, 4566.78 tokens/s
```

## Caveats

This run used only the default prompt. It is not prompt-robust evidence. Earlier
tiny smoke runs showed that reordered/xml prompt variants can fail the base
screen validity gate at small prompt counts, so prompt wording remains an
experimental variable until the screen is repeated with robust prompt variants
and enough examples.

The default `ENSEMBLE_KS` previously included `16`, so a P16 panel promoted all
candidates on holdout. That made this run a fuller audit than intended, but it
is too expensive for future smoke defaults. The script default now stops at
`1,4,8`; set `ENSEMBLE_KS=1,4,8,16` explicitly when a full-panel P16 holdout is
desired.

## Verdict

Do not claim the LoRA/spectral family is as powerful as dense Gaussian yet. The
next high-leverage run is not a larger blind scale-up; it is a prompt-robust,
multi-seed confirmation where vLLM shortlists a small `k`, PEFT confirms that
shortlist, and dense/spectral quality is judged on matched heldout panels.
