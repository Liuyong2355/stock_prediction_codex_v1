# Phase D4：LightGBM Rank Regression

唯一变量：D0 XGBoost Regressor 替换为冻结 E004 LightGBM Regressor；无调参、postprocess 或融合。

| Metric | D0 mean | D0 worst | D0 std | D4 mean | D4 worst | D4 std | Mean Δ | Improved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ic_mean | 0.120835 | 0.107731 | 0.009281 | 0.120923 | 0.109746 | 0.007905 | +0.000088 | 1/3 |
| annual_excess | 0.477742 | 0.367726 | 0.080955 | 0.450875 | 0.371963 | 0.065931 | -0.026867 | 1/3 |
| mean_turnover | 0.838435 | 0.858708 | 0.014354 | 0.825267 | 0.832988 | 0.006631 | -0.013168 | 2/3 |
| final_score | 0.240126 | 0.195798 | 0.032055 | 0.236052 | 0.210448 | 0.020560 | -0.004074 | 1/3 |
| diagnostic_turnover | 0.844471 | 0.864197 | 0.014374 | 0.841745 | 0.858489 | 0.012525 | -0.002726 | 3/3 |
| missing_top_fraction | 0.000250 | 0.000262 | 0.000011 | 0.074918 | 0.195344 | 0.085404 | +0.074668 | 0/3 |

## Per-fold delta

| Metric | F1 | F2 | F3 | Mean |
|---|---:|---:|---:|---:|
| ic_mean | -0.001332 | +0.002015 | -0.000418 | +0.000088 |
| annual_excess | -0.026848 | +0.004238 | -0.057989 | -0.026867 |
| mean_turnover | +0.003783 | -0.041911 | -0.001377 | -0.013168 |
| final_score | -0.009722 | +0.014651 | -0.017151 | -0.004074 |
| diagnostic_turnover | -0.000484 | -0.005708 | -0.001986 | -0.002726 |
| missing_top_fraction | +0.022433 | +0.195093 | +0.006479 | +0.074668 |

## Complementarity

| Diagnostic | F1 | F2 | F3 | Mean |
|---|---:|---:|---:|---:|
| prediction_spearman | 0.754108 | 0.722704 | 0.884431 | 0.787081 |
| top10_jaccard | 0.604087 | 0.483122 | 0.571556 | 0.552922 |
| top10_overlap | 0.747404 | 0.633772 | 0.722424 | 0.701200 |
| top10_exclusive_union_fraction | 0.395913 | 0.516878 | 0.428444 | 0.447078 |
| top10_per_model_exclusive_fraction | 0.252596 | 0.366228 | 0.277576 | 0.298800 |

- Stronger main model: `False`
- Second model candidate: `False`
- Decision: `stop_plain_lightgbm_regression`
