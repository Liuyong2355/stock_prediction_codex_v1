# Phase D2-A validation

- Scope: one variable only, the D0 Candidate training target.
- Target: same-date finite-label percentile rank with average ties; `T(u)-0.5`,
  threshold `0.9`, fixed stretch coefficient `10`.
- Controls: the D0 162-feature view, E006 XGBoost parameters, F1/F2/F3,
  one-trading-day purge, seed 42, no postprocess, and the unmodified organizer
  evaluator were retained.
- Target tests confirm that every `u <= 0.9` value is byte-identical to the
  frozen `u-0.5` target, ties remain equal, daily ordering is monotone, and
  non-finite labels/alternate coefficients are rejected.
- Full test suite: `219 passed`.
- Saved-prediction replay: F1/F2/F3 reproduced `ic_mean`, `annual_excess`,
  `mean_turnover`, and `final_score` at absolute tolerance `1e-14` using
  round-trip float parsing.
- Official evaluator SHA-256:
  `72dc59987e92ba8ae6b506e114dfe8591b5608c9beccd0769d5d1f778f9189c2`.
- Model and prediction artifacts are retained locally under the stage folds and
  remain Git-ignored; committed fold JSON files contain complete metrics and
  provenance.
