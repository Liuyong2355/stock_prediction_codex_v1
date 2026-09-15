# Provisional evaluator implementation notes

This is an explicit implementation of unresolved PDF edges, **not a change to the competition specification and not an organizer-verified official evaluator**. The Python `POLICY` object is copied into every fold result. The official script was absent at implementation start and supplied during execution; the user has now confirmed its provenance. It is archived unchanged. See ../outputs/official_evaluator_comparison.md and evaluator_arrival_review.md for executed differences. Existing provisional results remain unchanged; frozen-rule conflicts require explicit versioned review.

## Preserved specified rules

- Daily Rank IC is Spearman correlation after excluding unavailable labels. It does not exclude limit-up/down stocks.
- Top excess excludes limit-up and unavailable labels; excess is the Top1 mean minus the mean of the remaining eligible universe, annualized by arithmetic daily mean times 252.
- Turnover excludes limit-up, **never excludes missing labels or limit-down**. Its Top1 set is independently selected, so it can differ from the return Top1 set. Compute the Jaccard distance between consecutive observed dates within the same fold, beginning at date two.
- Score = 0.40 * RankIC + 0.30 * annualized excess + 0.30 * (1 - turnover). No normalization.
- Auxiliary IC standard deviation uses the user-frozen ddof=0.

## Provisional, unresolved choices used for this run

1. Nonfinite predictions are excluded per metric and counted, never filled or rewritten. E000 therefore has a different eligible population from models where it produces NaN. Stored predictions retain those NaNs. Finite label checks also exclude infinity; audited labels have none.
2. Sort by descending pred and then ascending ts_code for deterministic, label-independent ties. Split ordered eligible rows into 10 groups with NumPy array_split: earlier groups receive remainder rows. Top1 is the first group and Bottom1 the last.
3. At least 10 eligible rows are required for return groups or a turnover set. Otherwise that metric/date is undefined (NaN), not zero.
4. Spearman requires at least two finite pairs and nonconstant predictions and labels. Average ranks for ties are handled by SciPy. Undefined daily IC remains NaN.
5. Aggregate each metric over its finite daily values and report coverage counts. IC positive fraction uses defined IC dates. Empty metric series and zero-std ICIR are NaN.
6. If either adjacent turnover set is undefined, that pair is NaN; do not bridge over the invalid date. Empty-union turnover is undefined.
7. Auxiliary Top absolute and Top-minus-Bottom returns use the same return eligibility and arithmetic annualization by 252.

These assumptions cannot resolve the official edge semantics. Results and rankings may change after official parity verification. No prediction smoothing or top-set persistence bonus is applied.

## B2 model orchestration

F1/F2/F3 and one observed training-date purge come directly from folds_v1.yaml. Purge precedes removal of nonfinite training labels. All validation rows receive a prediction, including those without labels; the evaluator applies its own metric-specific masks. Existing causal Basic40 features retain historical warm-up, including the purged date as past observable input.

E001 calls sklearn.linear_model.Ridge(alpha=10.0), with other defaults from the pinned sklearn version. Each fold drops columns entirely NaN in its supervised training rows, calls sklearn SimpleImputer(strategy='median') separately per retained column, then sklearn StandardScaler.partial_fit over **training-only batches**. Partial fitting changes memory use, not the train population or scaling definition. Model fitting uses sklearn's default solver; no training or linear algebra solver is reimplemented.

E002 calls lightgbm.LGBMRegressor with exactly the frozen model params. All 800 requested rounds use training data only; there is no validation eval_set, early stopping, target conversion or parameter search. Training matrices are float64; LightGBM handles feature NaNs natively. Each full experiment/fold runs sequentially in a fresh process to limit memory. Predictions are made in batches with frozen preprocessing.

Per-fold files contain compressed full validation predictions, daily metrics, split/purge details, required aggregate metrics, coverage, preprocessing state, exact resolved library parameters, timings, peak process working-set memory, library versions and input/artefact hashes. Models and large predictions remain local with tracked fingerprints. `outputs/experiment_log.csv` follows the frozen required log schema. E000's train_rows describes the available split; supervised_fit_rows=0 explicitly records that no model was fitted.
