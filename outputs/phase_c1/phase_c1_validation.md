# Phase C1 验证记录

- Git HEAD：`fe517e4e5cd7ec8662205a25219dbf315643be4e`。
- w=1/w=0 三折逐指标精确复现 E005/E006 C0b selected。
- daily rank 后只调用一次共享 EMA+hysteresis；alpha/exit 固定且未搜索。
- F3 不参与权重选择，仅在锁定后作 robustness 报告。
- C0/C0b、官方 evaluator 和输入预测运行前后指纹一致。
- 完整测试集：220 passed。
