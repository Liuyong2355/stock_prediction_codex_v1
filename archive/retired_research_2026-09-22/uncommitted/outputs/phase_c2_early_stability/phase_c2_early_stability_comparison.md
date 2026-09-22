# C2 early stability：2020 representation check

Primary comparison uses raw predictions; fixed C0b postprocess is secondary only. 2021 was not used.

| Candidate | Variant | Rank IC | Annual excess | Turnover | Score | Diagnostic turnover | Missing Top |
|---|---|---:|---:|---:|---:|---:|---:|
| C2-R1 | raw | 0.113044 | 0.468695 | 0.540708 | 0.323613 | 0.777694 | 100.00% |
| C2-R1 | postprocessed | 0.091664 | 0.291833 | 0.007836 | 0.421865 | 0.395013 | 99.96% |
| C2-R2 | raw | 0.113186 | 0.407570 | 0.556346 | 0.300642 | 0.789083 | 100.00% |
| C2-R2 | postprocessed | 0.093044 | 0.291013 | 0.007782 | 0.422187 | 0.459249 | 99.99% |

## Raw R2 minus R1

- ΔIC: +0.000143
- Δannual excess: -0.061125
- Δturnover: +0.015638
- ΔScore: -0.022972
