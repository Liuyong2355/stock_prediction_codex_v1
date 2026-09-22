# A-share next-day ranking research

正式基线：`4c0aa59898156a6c53721e4d1645ee1a766417af`。

## 当前研究结论

Phase D0 已固定 E006 的 XGBoost、Rank target、F1/F2/F3、参数与官方 evaluator，比较：

1. 原 Full147；
2. Full147 加入更多现有基础因子的同日 percentile-rank 表示。

映射队友 36 个 core factors 后，在 Full147 中得到 30 个可比来源；其中 15 个已有
`csr_*`，实际只新增 15 个同日 percentile-rank 特征。E006 的 Rank IC、Top10% excess
和 Official Score 均为三折全胜，平均增量分别为 `+0.002408`、`+0.020199` 和
`+0.006269`，missing-label Top fraction 仅 `0.02%`。

Phase D1 将完全相同的 feature view 放入 corrected LambdaRank。Rank IC 三折全胜，
平均 `+0.002273`；excess 与 Score 仅 F2 改善，平均 Score `+0.003845`。因此该视图已被
确认是稳定的排序表示增量，但不是稳定的 LambdaRank tail/Score 增量。停止继续扩展
rank allowlist、技术指标和 turnover 参数。

## 冻结基线

- E006（Full147 + XGBoost + Rank）是当前干净 alpha baseline。
- corrected LambdaRank 核心协议保留为条件性第二阶段候选。
- E007、turnover 网格扩展、E005/E006 fusion、C2-R2 全量替换视图及已完成的早期消融
  均已停止，不再从默认入口运行。
- 历史结果仍保留在 `outputs/`；退役配置、测试和未提交研究文件保存在
  `archive/retired_research_2026-09-22/`。

## 入口

- 项目约束：`AGENTS.md`
- 文档索引：`docs/index.md`
- 冻结决策：`docs/DECISIONS.md`
- E006/E007 证据：`outputs/e006_e007_comparison.md`
- corrected LambdaRank 证据：`outputs/phase_c2/phase_c2_comparison.md`
- E006 rank-view 结果：`outputs/phase_d0_e006_rank_view/comparison.md`
- corrected LambdaRank rank-view 结果：`outputs/phase_d1_lambdarank_rank_view/comparison.md`
- 历史产物索引：`outputs/README.md`

所有 rolling/lag 必须按 `ts_code` 且按日期升序计算；同日截面特征只能使用当日可见
信息；验证必须使用冻结的 expanding-window folds 与一交易日 purge。
