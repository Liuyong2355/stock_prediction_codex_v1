# C2 R1 feature-family interaction

Raw prediction only; 2021 is unused. The 137-feature anchor is reused after strict validation.

| Feature set | Fold | IC | Excess | Turnover | Score | Diagnostic turnover | Missing Top | ΔIC | Δexcess | Δturnover | ΔScore |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| minus_market | A2019 | 0.140068 | 0.680247 | 0.524120 | 0.402865 | 0.789820 | 100.00% | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| minus_market | B2020 | 0.114573 | 0.509292 | 0.541154 | 0.336270 | 0.834246 | 100.00% | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| minus_market_relative | A2019 | 0.140345 | 0.667369 | 0.534715 | 0.395934 | 0.777895 | 100.00% | +0.000277 | -0.012878 | +0.010595 | -0.006931 |
| minus_market_cross_section_rank | A2019 | 0.137739 | 0.652956 | 0.500674 | 0.400780 | 0.768557 | 99.99% | -0.002329 | -0.027290 | -0.023445 | -0.002085 |
| minus_market_relative_cross_section_rank | A2019 | 0.135556 | 0.621494 | 0.525456 | 0.383034 | 0.798836 | 99.49% | -0.004512 | -0.058753 | +0.001337 | -0.019832 |
| minus_market_relative | B2020 | 0.115193 | 0.526005 | 0.539732 | 0.341959 | 0.846347 | 100.00% | +0.000620 | +0.016713 | -0.001422 | +0.005688 |
| minus_market_cross_section_rank | B2020 | 0.113847 | 0.505110 | 0.540449 | 0.334937 | 0.814580 | 100.00% | -0.000726 | -0.004181 | -0.000705 | -0.001333 |
| minus_market_relative_cross_section_rank | B2020 | 0.112609 | 0.494071 | 0.551067 | 0.327945 | 0.834444 | 100.00% | -0.001963 | -0.015221 | +0.009913 | -0.008326 |

Selected by the frozen rule: `minus_market`.
