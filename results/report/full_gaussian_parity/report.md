# Gaussian vs LoRA Parity Audit

## Capacity Summary

- Total dense parameters: `169869312`
- Total LoRA parameters: `1843200`
- LoRA parameter fraction: `0.010851`
- Summed dense rank almost surely: `82944`
- Summed LoRA rank cap: `576`
- Summed rank fraction: `0.006944`

A low-rank LoRA perturbation cannot exactly represent an arbitrary dense Gaussian perturbation unless the dense perturbation's rank is at most the LoRA rank.

## Empirical Projection Samples

### Shape `128x128`

| rank | captured Frobenius energy | relative Frobenius error |
| ---: | ---: | ---: |
| 1 | 0.030733 | 0.984513 |
| 2 | 0.059593 | 0.969746 |
| 4 | 0.112963 | 0.941826 |
| 8 | 0.210588 | 0.888489 |
| 16 | 0.376133 | 0.789853 |
| 32 | 0.625375 | 0.612067 |
| 64 | 0.895293 | 0.323585 |

Required rank by energy threshold: 0.50: r=24, 0.90: r=65, 0.99: r=99

### Shape `256x128`

| rank | captured Frobenius energy | relative Frobenius error |
| ---: | ---: | ---: |
| 1 | 0.022640 | 0.988615 |
| 2 | 0.044216 | 0.977642 |
| 4 | 0.085283 | 0.956408 |
| 8 | 0.160525 | 0.916229 |
| 16 | 0.293919 | 0.840287 |
| 32 | 0.508642 | 0.700969 |
| 64 | 0.792315 | 0.455725 |

Required rank by energy threshold: 0.50: r=32, 0.90: r=85, 0.99: r=119
