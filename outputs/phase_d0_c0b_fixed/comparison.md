# D0 + frozen C0b

F1/F2 tuned C0b policy transfer; F3 also previously exposed. All folds exploratory, not independent holdouts.

No model retraining or parameter search. Official organizer evaluator replayed unchanged.

| Fold | Strategy | IC | Top10 excess | Official turnover | Score | Diagnostic turnover | Missing-label Top |
|---|---|---:|---:|---:|---:|---:|---:|
| F1 | D0 raw | 0.128035 | 0.560191 | 0.829205 | 0.270510 | 0.838861 | 0.03% |
| F1 | D0+C0b | 0.103913 | 0.366036 | 0.209136 | 0.388635 | 0.279476 | 0.03% |
| F1 | E006+C0b | 0.102634 | 0.367332 | 0.212936 | 0.387372 | 0.282422 | 0.03% |
| F2 | D0 raw | 0.107731 | 0.367726 | 0.858708 | 0.195798 | 0.864197 | 0.03% |
| F2 | D0+C0b | 0.090492 | 0.169931 | 0.216312 | 0.322282 | 0.264826 | 0.02% |
| F2 | E006+C0b | 0.088442 | 0.169381 | 0.221121 | 0.319855 | 0.269362 | 0.02% |
| F3 | D0 raw | 0.126739 | 0.505310 | 0.827393 | 0.254071 | 0.830355 | 0.02% |
| F3 | D0+C0b | 0.104906 | 0.300669 | 0.206045 | 0.370349 | 0.230686 | 0.02% |
| F3 | E006+C0b | 0.102951 | 0.289599 | 0.198300 | 0.368570 | 0.222939 | 0.02% |

## Three-fold mean / worst / std

| Strategy | Metric | Mean | Worst | Std |
|---|---|---:|---:|---:|
| raw | ic_mean | 0.120835 | 0.107731 | 0.009281 |
| raw | annual_excess | 0.477742 | 0.367726 | 0.080955 |
| raw | mean_turnover | 0.838435 | 0.858708 | 0.014354 |
| raw | final_score | 0.240126 | 0.195798 | 0.032055 |
| raw | diagnostic_turnover | 0.844471 | 0.864197 | 0.014374 |
| raw | missing_top_fraction | 0.000250 | 0.000262 | 0.000011 |
| D0_C0b | ic_mean | 0.099770 | 0.090492 | 0.006573 |
| D0_C0b | annual_excess | 0.278879 | 0.169931 | 0.081529 |
| D0_C0b | mean_turnover | 0.210498 | 0.216312 | 0.004301 |
| D0_C0b | final_score | 0.360422 | 0.322282 | 0.027983 |
| D0_C0b | diagnostic_turnover | 0.258329 | 0.279476 | 0.020441 |
| D0_C0b | missing_top_fraction | 0.000223 | 0.000262 | 0.000038 |
| delta | ic_mean | -0.021065 | -0.024122 | 0.002862 |
| delta | annual_excess | -0.198863 | -0.204641 | 0.004347 |
| delta | mean_turnover | -0.627937 | -0.620069 | 0.010237 |
| delta | final_score | 0.120296 | 0.116279 | 0.004440 |
| delta | diagnostic_turnover | -0.586142 | -0.559385 | 0.018920 |
| delta | missing_top_fraction | -0.000027 | 0.000000 | 0.000038 |

## Interpretation

D0+C0b mean Score exceeds D0 raw by +0.120296, but exceeds E006+C0b (0.358599) by only +0.001823. All three fold Score differences against E006+C0b are positive but below 0.003.

Score decomposition versus D0 raw: IC -0.008426, Top10 excess -0.059659, official turnover +0.188381.

The gain comes from lower turnover despite a substantial loss of IC and Top10 excess. Diagnostic turnover also falls; missing-label Top stays negligible. F2 Top10 excess is only 0.169931.

The predeclared meaningful +0.01 Score margin versus E006+C0b is not met. Retain D0 raw as primary Alpha and E006+C0b as low-missing high-Score reference; do not promote D0+C0b or reopen turnover tuning.
