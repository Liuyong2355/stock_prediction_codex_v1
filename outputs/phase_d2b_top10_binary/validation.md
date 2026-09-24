# Phase D2-B validation

- Scope: one task change only, from D0 continuous rank regression to the fixed
  same-date Top10% binary target.
- Labels: finite supervised rows after the frozen one-day purge; average-tie
  daily percentile rank; positive iff `u > 0.9`. Fold positive fractions are
  `10.0091%`, `10.0094%`, and `10.0087%`.
- Model: `XGBClassifier`; all E006 XGBoost parameters retained except the
  required objective change from `reg:squarederror` to `binary:logistic`.
  Positive-class probabilities are passed directly to evaluation.
- Controls: D0 162-feature view, F1/F2/F3, training samples, purge, seed 42,
  official evaluator, and no postprocess are unchanged. No class weights,
  parameter search, probability calibration, threshold variation, or fusion.
- Full test suite: `224 passed`.
- Saved-prediction replay: F1/F2/F3 reproduced `ic_mean`, `annual_excess`,
  `mean_turnover`, and `final_score` with the unmodified organizer evaluator
  using round-trip float parsing.
- Complementarity replay: per-fold daily prediction Spearman, Top10 Jaccard,
  and Top10 overlap were independently recomputed from saved D2-B and D0
  predictions and matched the result JSON at absolute tolerance `1e-14`.
- Official evaluator SHA-256:
  `72dc59987e92ba8ae6b506e114dfe8591b5608c9beccd0769d5d1f778f9189c2`.
- Local model and prediction artifacts remain Git-ignored. Fold result JSON and
  daily complementarity CSV files retain the committed evidence.
