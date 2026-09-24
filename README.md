# A-share next-day ranking project

The canonical primary raw alpha model is **`main_alpha_xgb_rank162`**
(historical experiment alias: D0). It is the frozen XGBoost Rank regressor using
Full147 plus 15 same-date percentile-rank views. Its F3 artifact, feature order,
hash and training cutoff are in [config/main_model.yaml](config/main_model.yaml);
the load guard is in `src/stock_prediction/main_model.py`.

Across 2022/2023/2024 validation folds, this model's mean Rank IC is `0.120835`,
mean Top10 annual excess is `0.477742`, and mean Official Score is `0.240126`.
The once-scored 2025–2026 local test period gives Rank IC `0.129418`, Top10
excess `0.461515`, and Score `0.238108`. No test labels were used to train it.

**Model signal and full-strategy Score are different comparisons.** Historical
E005/E006 strategies with the frozen C0b turnover postprocess achieve higher
Official Scores by reducing turnover. E005+C0b has the highest later-test Score
in the checked set (`0.369300`) but had severe missing-label Top occupancy in
validation. E006+C0b scores `0.366424` on that test with negligible missing
Top occupancy; it is documented as a frozen low-missing scoring strategy, not
the primary 162-feature raw model. See
[the model audit](docs/MODEL_AUDIT_2026-09-24.md) for all E-series, F-fold and
test-period evidence.

## Where to start

- [Project rules](AGENTS.md) and [documentation index](docs/index.md)
- [Main model identity](docs/MAIN_MODEL.md) and [frozen configuration](config/main_model.yaml)
- [Current handoff](docs/NEXT_RESEARCH_HANDOFF.md)
- [Research decisions](docs/DECISIONS.md)
- [Completed research archive](archive/completed_research_2026-09-24/README.md)

Feature matrices and raw test X/Y remain local. The canonical F3 model binary is
committed; historical predictions and model binaries remain local unless
otherwise specified. The local test labels are reconstructed after the fact
from next-day close and must not become a new tuning fold.

To verify the current codebase in the project environment:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
