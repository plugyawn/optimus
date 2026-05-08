# Search Quality Confirmation

Gate: **PASS**

| metric | value |
| --- | ---: |
| dense best screen exact | 0.09375 |
| confirmed best screen exact | 0.078125 |
| screen delta vs dense | -0.015625 |
| dense best strict holdout | 0.0546875 |
| confirmed best strict holdout | 0.078125 |

## Rows

| k | confirmed strict | dense strict at k | delta vs dense best | delta vs same k | full speedup | pass quality | pass speed |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 0.03125 | 0.0390625 | -0.0234375 | -0.0078125 | 6.21051 | false | true |
| 4 | 0.0703125 | 0.046875 | 0.015625 | 0.0234375 | 4.92387 | true | true |
| 8 | 0.078125 | 0.0546875 | 0.0234375 | 0.0234375 | 4.00771 | true | true |

## Gate Checks

| check | pass | detail |
| --- | --- | --- |
| dense_validity_pass | true | `{"pass": true}` |
| confirmed_validity_pass | true | `{"pass": true}` |
| strict_holdout_quality_at_speed | true | `{"max_confirm_k": 8, "min_full_speedup": 1.0, "min_holdout_delta": 0.0, "passing_k": 4}` |
