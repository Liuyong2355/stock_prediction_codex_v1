# C2 R1 feature-family ablation — round 1

Raw prediction only. 2019 and 2020 are the only OOT validation years; 2021 is unused.

| Feature set | Fold | IC | Excess | Turnover | Score | Diagnostic turnover | Missing Top | ΔIC | Δexcess | Δturnover | ΔScore |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full147 | A2019 | 0.133743 | 0.587750 | 0.504499 | 0.378472 | 0.781571 | 99.96% | — | — | — | — |
| minus_cross_section_rank | A2019 | 0.131601 | 0.601183 | 0.510716 | 0.379781 | 0.729075 | 99.96% | -0.002142 | +0.013433 | +0.006217 | +0.001308 |
| minus_relative | A2019 | 0.134551 | 0.598719 | 0.500144 | 0.383393 | 0.787603 | 99.97% | +0.000808 | +0.010969 | -0.004354 | +0.004920 |
| minus_market | A2019 | 0.140068 | 0.680247 | 0.524120 | 0.402865 | 0.789820 | 100.00% | +0.006325 | +0.092497 | +0.019621 | +0.024393 |
| Full147 | B2020 | 0.113044 | 0.468695 | 0.540708 | 0.323613 | 0.777694 | 100.00% | — | — | — | — |
| minus_cross_section_rank | B2020 | 0.114594 | 0.501922 | 0.524204 | 0.339153 | 0.809283 | 100.00% | +0.001551 | +0.033227 | -0.016504 | +0.015540 |
| minus_relative | B2020 | 0.116741 | 0.500359 | 0.550995 | 0.331505 | 0.778252 | 100.00% | +0.003697 | +0.031664 | +0.010286 | +0.007892 |
| minus_market | B2020 | 0.114573 | 0.509292 | 0.541154 | 0.336270 | 0.834246 | 100.00% | +0.001529 | +0.040597 | +0.000445 | +0.012657 |

## Frozen-rule decisions

| Ablation | Decision |
|---|---|
| minus_cross_section_rank | deletion_candidate |
| minus_market | deletion_candidate |
| minus_relative | deletion_candidate |
