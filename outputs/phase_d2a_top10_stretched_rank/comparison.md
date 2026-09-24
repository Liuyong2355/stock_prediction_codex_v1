# Phase D2-A：Top10-stretched continuous rank target

唯一变量：训练目标由 `u-0.5` 改为 `T(u)-0.5`；特征、模型参数、fold、purge、seed、evaluator 与无 postprocess 均保持 D0 不变。

| Metric | D0 mean | D0 worst | D0 std | D2-A mean | D2-A worst | D2-A std | Mean Δ | Improved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ic_mean | 0.120835 | 0.107731 | 0.009281 | 0.080186 | 0.053007 | 0.019221 | -0.040649 | 0/3 |
| annual_excess | 0.477742 | 0.367726 | 0.080955 | 0.398918 | 0.155559 | 0.173388 | -0.078824 | 0/3 |
| mean_turnover | 0.838435 | 0.858708 | 0.014354 | 0.833579 | 0.843282 | 0.009305 | -0.004856 | 1/3 |
| final_score | 0.240126 | 0.195798 | 0.032055 | 0.201676 | 0.121562 | 0.056907 | -0.038450 | 0/3 |
| diagnostic_turnover | 0.844471 | 0.864197 | 0.014374 | 0.836369 | 0.846356 | 0.009202 | -0.008102 | 1/3 |
| missing_top_fraction | 0.000250 | 0.000262 | 0.000011 | 0.007163 | 0.014958 | 0.005783 | +0.006914 | 0/3 |

## Per-fold delta

| Metric | F1 | F2 | F3 | Mean |
|---|---:|---:|---:|---:|
| ic_mean | -0.034653 | -0.054724 | -0.032570 | -0.040649 |
| annual_excess | -0.013570 | -0.212167 | -0.010736 | -0.078824 |
| mean_turnover | +0.014077 | -0.037678 | +0.009032 | -0.004856 |
| final_score | -0.022155 | -0.074236 | -0.018958 | -0.038450 |
| diagnostic_turnover | +0.007495 | -0.040047 | +0.008245 | -0.008102 |
| missing_top_fraction | +0.014696 | +0.000871 | +0.005174 | +0.006914 |

## Score delta decomposition

- IC: -0.016260
- Top10% excess: -0.023647
- Official turnover: +0.001457
- Structural breakthrough candidate: `False`
