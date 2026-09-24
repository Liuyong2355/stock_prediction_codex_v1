# Locked competition submission — E006+C0b

The local submission is `../submissions/E006_C0b_fixed/submission.csv`. It is
ready for user upload but has **not** been sent to the organizer. The large CSV
is ignored by Git; the small `manifest.json` beside it records its SHA-256:
`e6dfcdf3ec58b0ea7bac5c3e044008c4e7f637d83c5ccb0890277a93c2cee30b`.

## Frozen choice

- E006 Full147 XGBoost Rank-regression F3 model, 800 rounds, trained only
  through `2023-12-28` after the one-day purge. No refit or model search.
- Existing C0b daily-rank causal EMA `alpha=0.30` and Top10 hysteresis
  `exit_fraction=0.25`; no parameter adjustment or ensemble.
- Selection rationale: E006+C0b is the established low-missing high-Score
  reference. D0+C0b's small incremental three-fold Score edge (`0.001823`)
  did not pass its promotion gate; E005/C2 have historical missing-label
  selection anomalies. Do not use already-viewed reconstructed test Y to
  reverse this choice.

## Construction and checks

The input test X, Full147 matrix and keys, frozen E006 F3 model, saved
2025-01-02–2026-06-05 raw predictions, and C0b decision were hash-checked.
Only the final date, `2026-06-08` (4,650 rows), was newly predicted from the
frozen model and same-date causal features. C0b was replayed across all 344
test dates. Its first 343 dates matched the saved C0b predictions exactly.
No test Y or label-derived score was read during construction.

The resulting file has exactly the required `ts_code,trade_date,pred` columns,
all 1,599,600 test X keys in the original order, no duplicates or nonfinite
scores, and 4,650 distinct scores on each date. Scores are ordinal ranking
values, not calibrated return estimates; the official evaluator uses only
their ordering for all three weighted metrics. The final date is not locally
scorable because no subsequent close is supplied. Historical frozen
E006+C0b Score is not a claim about this complete submitted file or an
organizer-issued test result.

Reproduction entry point: `python -m stock_prediction.build_competition_submission`
with `src` on `PYTHONPATH`. The entry point deliberately refuses to overwrite
an existing submission. To rebuild, use a separate clean workspace and verify
the resulting SHA-256 against the manifest; do not remove the locked copy.
`--verify-existing` reconstructs all predictions in memory from the dedicated
no-label lock config and checks the local CSV exactly; this check has passed.
