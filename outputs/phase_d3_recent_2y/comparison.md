# Phase D3：Recent-2Y training window

唯一变量：训练历史由 D0 expanding history 改为验证年前最近两个自然年；其余条件不变。无调参、postprocess 或融合。

| Metric | D0 mean | D0 worst | D0 std | Recent-2Y mean | Recent-2Y worst | Recent-2Y std | Mean Δ | Improved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ic_mean | 0.120835 | 0.107731 | 0.009281 | 0.104569 | 0.091537 | 0.009229 | -0.016266 | 0/3 |
| annual_excess | 0.477742 | 0.367726 | 0.080955 | 0.419696 | 0.318409 | 0.075524 | -0.058046 | 0/3 |
| mean_turnover | 0.838435 | 0.858708 | 0.014354 | 0.838705 | 0.845893 | 0.005294 | +0.000269 | 1/3 |
| final_score | 0.240126 | 0.195798 | 0.032055 | 0.216125 | 0.181062 | 0.025384 | -0.024001 | 0/3 |
| diagnostic_turnover | 0.844471 | 0.864197 | 0.014374 | 0.844237 | 0.853843 | 0.007270 | -0.000234 | 1/3 |
| missing_top_fraction | 0.000250 | 0.000262 | 0.000011 | 0.000289 | 0.000316 | 0.000020 | +0.000039 | 0/3 |

## Per-fold delta

| Metric | F1 | F2 | F3 | Mean |
|---|---:|---:|---:|---:|
| ic_mean | -0.017571 | -0.016194 | -0.015034 | -0.016266 |
| annual_excess | -0.060499 | -0.049317 | -0.064323 | -0.058046 |
| mean_turnover | +0.016688 | -0.021789 | +0.005908 | +0.000269 |
| final_score | -0.030184 | -0.014736 | -0.027083 | -0.024001 |
| diagnostic_turnover | +0.014982 | -0.021588 | +0.005903 | -0.000234 |
| missing_top_fraction | +0.000054 | +0.000027 | +0.000036 | +0.000039 |

## Complementarity

| Diagnostic | F1 | F2 | F3 | Mean |
|---|---:|---:|---:|---:|
| prediction_spearman | 0.862135 | 0.735554 | 0.760666 | 0.786118 |
| top10_jaccard | 0.412757 | 0.287517 | 0.303490 | 0.334588 |
| top10_overlap | 0.574397 | 0.438927 | 0.455231 | 0.489518 |

- Stronger main model: `False`
- Second alpha candidate: `True`
- Decision: `retain_recent_2y_without_fusion`
