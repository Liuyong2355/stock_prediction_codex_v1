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

Phase D2-A 在 D0 Candidate 上仅将训练 rank target 的真实收益 Top10% 区域按固定系数
10 拉伸。Rank IC、Top10% excess 和 Official Score 均三折全败，平均 Score 从
`0.240126` 降至 `0.201676`。该连续 target-stretch 路线停止，不搜索阈值、系数或新公式。

Phase D2-B 在相同 D0 条件下仅改为真实收益 Top10% 二分类任务。三折 Rank IC 与
Top10% excess 均为负，平均 excess 为 `-0.403016`。虽然预测与 D0 明显不同，但它是
稳定反向而非独立 tail alpha；missing-label Top 占比升至 `15.65%`。整个专门 Tail
target 研究路线正式停止，不做阈值、class weight、分类参数或融合扩展。

Phase D3 将 D0 的训练历史唯一改为验证年前最近两个自然年。Recent-2Y 的 IC、excess
和 Score 均三折低于 D0，Mean Score 为 `0.216125`，不替代主模型；但三折均保持正
IC、正 excess、正 Score，且与 D0 的平均 Top10 overlap 仅 `48.95%`，因此保留为第二
Alpha 候选。本阶段不融合，并停止所有其他窗口长度和时间衰减实验。

Phase D4 用冻结 E004 参数的 LightGBM Regressor 替换 D0 XGBoost。其三折 alpha
均为正，Mean Score `0.236052`，但仅 F2 超过 D0；Top10 overlap 为 `70.12%`，且
missing-label Top 占比均值 `7.49%`、F2 达 `19.53%`。因此不晋级为干净第二模型
候选，停止普通 LightGBM regression 路线，不调参、不融合。

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
- D2-A Top10-stretched target 结果：`outputs/phase_d2a_top10_stretched_rank/comparison.md`
- D2-B Top10 binary target 结果：`outputs/phase_d2b_top10_binary/comparison.md`
- D3 Recent-2Y 结果：`outputs/phase_d3_recent_2y/comparison.md`
- D4 LightGBM Rank Regression 结果：`outputs/phase_d4_lightgbm_rank_regression/comparison.md`
- 历史产物索引：`outputs/README.md`

所有 rolling/lag 必须按 `ts_code` 且按日期升序计算；同日截面特征只能使用当日可见
信息；验证必须使用冻结的 expanding-window folds 与一交易日 purge。

## 训练可视化

后续获准实验可使用 SwanLab 记录聚合指标与复现元数据。默认关闭，云端启用、登录方式
和禁止上传的数据范围见 `docs/SWANLAB.md`。SwanLab 不得改变训练或官方评分流程。
