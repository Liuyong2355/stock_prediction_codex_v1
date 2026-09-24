# Phase D4 validation

- Scope: one variable only, D0 XGBoost Regressor replaced by frozen E004
  `lightgbm_reg_v1` parameters.
- D0 features, Rank target, expanding folds, purge, validation rows, seed,
  evaluator, and no postprocess are unchanged. No tuning or fusion.
- The first F1 run completed evaluation but native LightGBM model saving rejected
  the Unicode path. It produced no result JSON. The existing Unicode-safe writer
  was used for an otherwise identical rerun.
- SwanLab cloud logging succeeded for all folds and the summary; only aggregate
  metrics and reproducibility metadata were uploaded.
- Full test suite: `232 passed`.
- Official saved-prediction replay and complementarity replay matched fold JSON
  at absolute tolerance `1e-14`.
- Official evaluator SHA-256:
  `72dc59987e92ba8ae6b506e114dfe8591b5608c9beccd0769d5d1f778f9189c2`.
- Models and predictions remain Git-ignored. No fusion was run.
