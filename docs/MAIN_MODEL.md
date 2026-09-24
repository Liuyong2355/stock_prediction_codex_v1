# Canonical main model

`main_alpha_xgb_rank162` is the production-facing name of the unchanged D0 F3
XGBoost artifact. The name clarifies that this is the primary **raw Alpha**
model rather than the historical winner of every Score comparison.

- Binary: `models/main_alpha_xgb_rank162/model.ubj` (committed to Git).
- Historical source: `outputs/phase_d0_e006_rank_view/F3/model.ubj`.
- SHA-256: `096f50d7b306be28b22a1645c95d5e9e536478fb9423f0e1b3da8d48a84066ac`.
- Features: frozen Full147 column order followed by 15 D0 rank-view columns,
  exactly 162 names in `config/main_model.yaml` and
  `config/phase_d0_e006_rank_view.yaml`.
- Training target: within-date average percentile rank of finite next-day
  return labels, minus 0.5. Seed and XGBoost parameters are those of frozen
  E006. No turnover postprocess.
- Training cutoff: 2023-12-28 after one-trading-day purge. F3 validates 2024;
  the later local test covers 2025-01-02 to 2026-06-05.

Use `stock_prediction.main_model.load_main_model(root)` to load the binary
and verify its hash, exact feature order and 800 boosting rounds. The
canonical name does not retrain or alter predictions. F1/F2 model artifacts
remain at the historical D0 paths for fold reproduction.

Raw predictions should not be compared directly with C0b postprocessed
strategies as a model-only contest: the latter trade some ordering and excess
return for much lower official turnover. The full comparison and decision
boundary are in `docs/MODEL_AUDIT_2026-09-24.md`.
