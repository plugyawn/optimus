# Dense Gaussian vs LoRA Parity Report

| metric | value |
| --- | ---: |
| shared candidates | 16 |
| Spearman | 0.008973031700969426 |
| top-8 overlap | 4 |
| selected regret | 0.03125 |
| dense candidate/sec | 0.21226869757424394 |
| LoRA candidate/sec | 0.2516969192094384 |
| speed ratio LoRA/dense | 1.185746754400299 |

## Gates

| gate | pass |
| --- | ---: |
| shared_panel | true |
| spearman | false |
| topk_overlap | false |
| selected_regret | false |
| speed | true |

Overall pass: `false`
