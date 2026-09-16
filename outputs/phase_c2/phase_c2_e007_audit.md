# Phase C2：E007 审计

No group/order/direction bug. E007 tested a materially different and under-specified ranker protocol; C2 isolates explicit linear gain, truncation 500, norm=true, 100 rounds, and corrected decile boundary formula.

- relevance 方向正确；日期 group、group sum、稳定排序均已验证。
- E007 使用 raw Full147、800 rounds；未显式固定线性 label_gain、truncation=500、norm=true。
- 旧 floor 标签在精确十分位边界与 C2 ceil-1 协议不同。
