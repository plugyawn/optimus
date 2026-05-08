# Family Sweep Report

Overall pass: `false`

## Arm Pass

| arm | pass |
| --- | ---: |
| sparse_d0p125 | false |
| sparse_d0p25 | false |

## Rows

| variant | arm | pass | ensemble delta examples vs baseline | arm ensemble | baseline ensemble | selected regret vs dense | Spearman vs dense | speed/dense | cap delta | malformed delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| default | lora | false | 0 | 0.078125 | 0.078125 | 0.03125 | -0.00869663 | 0.831909 | 0 | 0 |
| default | sparse_d0p125 | false | 2 | 0.09375 | 0.078125 | 0.0625 | -0.127321 | 0.799789 | 0 | 0 |
| default | sparse_d0p25 | false | 0 | 0.078125 | 0.078125 | 0.0625 | -0.109811 | 0.808134 | 0 | 0 |
