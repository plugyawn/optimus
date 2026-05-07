# Dense Gaussian vs LoRA Parity Report

| metric | value |
| --- | ---: |
| shared candidates | 64 |
| Spearman | -0.07105301492957512 |
| top-8 overlap | 1 |
| selected regret | 0.09375 |
| dense best cap-hit | 0.0 |
| dense best malformed | 0.046875 |
| selected LoRA cap-hit | 0.015625 |
| selected LoRA malformed | 0.015625 |
| dense candidate/sec | 0.21211323021571146 |
| LoRA candidate/sec | 0.20143579658984104 |
| speed ratio LoRA/dense | 0.9496616330107657 |

## Gates

| gate | pass |
| --- | ---: |
| shared_panel | true |
| spearman | false |
| topk_overlap | false |
| selected_regret | false |
| speed | false |

Overall pass: `false`
