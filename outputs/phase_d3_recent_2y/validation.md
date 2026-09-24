# Phase D3 validation

- Scope: one variable only, D0 training-history length.
- Windows: F1 `2020-01-02..2021-12-30` (485 dates), F2
  `2021-01-04..2022-12-29` (484 dates), and F3
  `2022-01-04..2023-12-28` (483 dates). The frozen fold split is applied first,
  so its final training date remains purged.
- Controls: D0 162-feature view, XGBoost regression parameters, continuous
  daily rank target, seed 42, validation rows, official evaluator, and no
  postprocess are unchanged.
- Full test suite: `226 passed`.
- Saved-prediction replay: F1/F2/F3 reproduced `ic_mean`, `annual_excess`,
  `mean_turnover`, and `final_score` with the unmodified organizer evaluator at
  absolute tolerance `1e-14` using round-trip float parsing.
- Complementarity replay: daily prediction Spearman, Top10 Jaccard, and Top10
  overlap were independently recomputed from saved Recent-2Y and D0 predictions
  and matched fold JSON at absolute tolerance `1e-14`.
- Official evaluator SHA-256:
  `72dc59987e92ba8ae6b506e114dfe8591b5608c9beccd0769d5d1f778f9189c2`.
- Local models and prediction files remain Git-ignored. No fusion or alternate
  training window was run.
