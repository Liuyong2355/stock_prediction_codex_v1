# Phase C0 验证记录

- Git HEAD：`4c0aa59898156a6c53721e4d1645ee1a766417af`。
- 官方 evaluator SHA-256：`72dc59987e92ba8ae6b506e114dfe8591b5608c9beccd0769d5d1f778f9189c2`。
- 六份输入预测均通过既有指纹、schema、唯一 validation keys、行数及 raw 官方指标重放。
- F1/F2 完整 18×2 网格已评分；F3 仅在参数冻结后评估 selected 与两个 controls。
- 受保护文件在运行前后逐字节一致；未重训，未修改 E000–E007、特征、fold、target 或官方 evaluator。
- 因果、未来扰动、fold 隔离、精确 Top、涨停/小池重置、tie、finite、确定性、官方实际 Top 一致与重生成一致均由测试覆盖。
- 完整测试集：212 passed。
