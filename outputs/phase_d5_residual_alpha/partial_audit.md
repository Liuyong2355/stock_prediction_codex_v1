# D5 rank-difference artifact check

A feature can correlate with rank(y) - rank(D0) simply because it correlates with D0. This audit reports its direct true-return IC and its partial rank correlation with true return after controlling for D0 prediction rank, computed within each date and finite feature/label pair.

| Region | Family | Fold | True-return IC | Feature-D0 IC | D0-return IC | Partial true-return IC given D0 |
|---|---|---|---:|---:|---:|---:|
| full | volatility | F1 | -0.058584 | -0.264728 | 0.126058 | -0.023417 |
| full | volatility | F2 | -0.063709 | -0.355104 | 0.105668 | -0.030414 |
| full | volatility | F3 | -0.049075 | -0.255136 | 0.123664 | -0.014442 |
| full | volume_price | F1 | -0.042158 | -0.247844 | 0.126262 | -0.011025 |
| full | volume_price | F2 | -0.038183 | -0.193862 | 0.105977 | -0.018337 |
| full | volume_price | F3 | -0.045227 | -0.251729 | 0.124191 | -0.014619 |
| middle60 | volatility | F1 | -0.030057 | -0.180879 | 0.053893 | -0.019191 |
| middle60 | volatility | F2 | -0.032718 | -0.228712 | 0.043934 | -0.023915 |
| middle60 | volatility | F3 | -0.015519 | -0.143804 | 0.046244 | -0.008521 |
| middle60 | volume_price | F1 | -0.015654 | -0.140081 | 0.053902 | -0.008264 |
| middle60 | volume_price | F2 | -0.018273 | -0.097607 | 0.043973 | -0.014068 |
| middle60 | volume_price | F3 | -0.019470 | -0.137927 | 0.046274 | -0.013326 |
| top20 | volatility | F1 | -0.000138 | 0.114951 | 0.052539 | -0.005244 |
| top20 | volatility | F2 | -0.019468 | 0.069655 | 0.044773 | -0.022275 |
| top20 | volatility | F3 | 0.024991 | 0.039506 | 0.060446 | 0.022641 |
| top20 | volume_price | F1 | 0.000483 | -0.012065 | 0.053126 | 0.001704 |
| top20 | volume_price | F2 | -0.002974 | 0.015256 | 0.045167 | -0.002902 |
| top20 | volume_price | F3 | -0.005755 | -0.031596 | 0.061280 | -0.001809 |

Partial rank correlation is a diagnostic, not a tradable model result. It is not a training or validation score for a second model.
