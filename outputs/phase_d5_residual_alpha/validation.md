# D5 Validation

- Frozen D0 prediction artifacts: hash-checked and row-aligned to the official folds.
- Labels: exact equality checked against the frozen label cache.
- Feature construction: Full147 plus the frozen 15 D0 same-date average percentile-rank views.
- Residual and regions: reconstructed independently for each date using finite labels only.
- Training/fusion/postprocess: none.
