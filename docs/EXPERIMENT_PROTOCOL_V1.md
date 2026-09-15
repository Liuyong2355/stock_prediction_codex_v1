# Experiment Protocol V1.0.1

**Canonical configs:** `../config/folds_v1.yaml`, `../config/experiments_v1.yaml`

## 1. Research objective

Optimize the organizer's final score:

```text
Score = 0.40 * RankIC
      + 0.30 * AnnualizedTopExcessReturn
      + 0.30 * (1 - Turnover)
```

Do not optimize RMSE as the project objective. RMSE may be logged diagnostically, but model selection is based on out-of-time competition metrics and stability.

## 2. Data preparation

1. Parse and sort by `ts_code`, `trade_date`.
2. Preserve the organizer raw fields unchanged in the raw-data layer.
3. Generate features causally.
4. Replace infinite derived values with NaN.
5. Exclude rows with NaN `y_ret_1d` from supervised fitting.
6. Do not use validation/test labels in preprocessing fitted on training data.
7. Tree models may consume NaNs natively. Ridge must fit its imputer and scaler on training rows only.

## 3. Validation design

No random split is permitted for official research experiments.

Use three expanding-window folds:

| Fold | Candidate training period | Validation period |
|---|---|---|
| F1 | 2018-01-02 to 2021-12-31 | calendar year 2022 |
| F2 | 2018-01-02 to 2022-12-31 | calendar year 2023 |
| F3 | 2018-01-02 to 2023-12-31 | calendar year 2024 |

### One-trading-day purge

Because `y_ret_1d(t)` contains `close(t+1)`, after selecting the candidate training period remove the final distinct training `trade_date` before fitting. This prevents a training label from consuming a price in the validation period.

Do not hard-code the purged calendar date. Remove the last observed trading date from the training slice.

### Validation feature warm-up

A validation feature at date t may use historical observations before validation starts because those observations would have been known in real time. Therefore causal features should be computed on the chronological panel before fold slicing.

## 4. Frozen V1 targets

### Raw target

```text
y_raw = y_ret_1d
```

### Daily rank target

```text
y_rank = rank_pct(y_ret_1d within trade_date, average ties) - 0.5
```

Only finite labels participate in that date's rank.

### Daily relative-return target

```text
y_relative = y_ret_1d - same-date median(y_ret_1d)
```

### Ranker relevance target

For E007, rank finite same-date y_ret_1d with average percentile ranks, then relevance = min(floor(rank_pct * 10), 9). Equal labels must have equal relevance.

## 5. Required evaluation outputs

Every fold must output at least:

- mean daily Rank IC;
- daily Rank IC standard deviation;
- ICIR;
- fraction of days with IC > 0;
- annualized Top1 excess return;
- mean turnover;
- final competition score;
- Top1 annualized absolute return;
- Top1-Bottom1 annualized spread.

The official organizer evaluator is authoritative when available.

## 6. Initial experiments

| ID | Features | Model | Target | Purpose |
|---|---|---|---|---|
| E000 | `ret_1` only | identity (`pred=ret_1`) | raw | metric/pipeline smoke test |
| E001 | Basic40 | Ridge | raw | linear baseline |
| E002 | Basic40 | LightGBM regression | raw | primary baseline |
| E003 | Full147 | LightGBM regression | raw | expanded-feature value |
| E004 | Full147 | LightGBM regression | rank | rank-target test |
| E005 | Full147 | LightGBM regression | relative | market-relative target test |
| E006 | Full147 | XGBoost regression | rank | second GBDT family |
| E007 | Full147 | LightGBM Ranker | rank decile | direct ranking experiment |

Model parameter values are frozen in `config/experiments_v1.yaml` for the baseline phase. They are not asserted to be optimal.

## 7. Mandatory implementation order

### Phase A — trustworthy baseline pipeline

Implement only:

1. raw-data audit;
2. Basic40;
3. F1/F2/F3 temporal split and purge;
4. evaluator;
5. E000;
6. E001;
7. E002;
8. experiment logging;
9. tests for leakage, feature count/names, split boundaries, evaluator parity when official evaluator is available.

Do not implement Full147 first.

### Phase B — full V1 features and target/model comparison

After Phase A passes:

1. Full147;
2. E003;
3. E004;
4. E005;
5. E006;
6. E007.

### Phase C — evidence-driven research

Only after E000-E007 are reproducible:

- single-factor diagnostics;
- feature-family ablations;
- target comparison;
- model comparison;
- ensemble research;
- turnover-aware post-processing.

## 8. Experiment logging

Append one row per experiment-fold run to `outputs/experiment_log.csv`.

The canonical required columns are in `config/experiments_v1.yaml`.

Never rely on memory or notebook-only outputs to compare models.

A run must record:

- experiment and fold identity;
- git commit;
- data/spec/protocol versions;
- exact params and seed;
- date ranges and row counts;
- all required metrics;
- time/memory usage;
- notes.

## 9. Model-selection rule

Do not select a model only because F3/2024 is best.

For every candidate summarize:

```text
Score_F1
Score_F2
Score_F3
MeanScore
WorstScore
StdScore
```

Preferred models have:

1. high mean out-of-time score;
2. acceptable worst-fold score;
3. reasonable cross-fold stability.

A large one-year spike with failure in other years is not sufficient evidence.

## 10. Feature-family ablation rule

After Full147 is stable, run controlled ablations such as:

- Full147 minus volatility;
- Full147 minus volume/amount;
- Full147 minus market state;
- Full147 minus cross-sectional ranks;
- Full147 minus limit features.

A feature family should be retained based on evidence across folds, not just LightGBM importance.

## 11. Turnover research

Turnover is 30% of the official score, so prediction stabilization is a dedicated research stage.

A first frozen family of post-processing experiments may use:

```text
p_final(t) = alpha * p_raw(t) + (1-alpha) * p_final(t-1)
```

Test a predeclared alpha grid only after base-model selection. Choose alpha by final out-of-time score, not IC alone.

Do not introduce holding/persistence bonuses until the simple smoothing baseline is understood.

## 12. Leakage checklist

Before trusting any score, confirm:

- no `shift(-k)` or centered rolling windows in features;
- target is never part of X;
- target-derived rank/relative transformations are used only as training labels;
- train-fitted imputers/scalers never see validation rows;
- no rolling operation crosses `ts_code`;
- the one-day label purge is applied;
- cross-sectional features use only same-date observables;
- test labels are never referenced.

## 13. Reproducibility

- Seed = 42 for V1 baseline experiments.
- Config files, not notebooks, define canonical experiments.
- Any material protocol change requires a decision record and version bump.

## 14. Frozen implementation clarifications (V1.0.1)

- Auxiliary daily IC standard deviation uses ddof=0.
- For Ridge, drop features that are entirely NaN in that fold's supervised training rows. Record their names in dropped_all_nan_features_json. Fit median imputation and scaling only on the remaining training data; apply the same column selection to validation.
- The organizer evaluate.py is absent. Any internal evaluator must be labeled provisional until parity is verified.
- Raw observations must not be calendar-reindexed, forward filled or backfilled. Per-stock lags retain observed-row semantics.
- No fold dates, purge length, baseline parameters, seed, or experiment sequence changed.
