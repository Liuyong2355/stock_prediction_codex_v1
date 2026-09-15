# Frozen Full147 and E003 implementation

## Feature generation

The 147 names, formulas and ordering come from features_v1.yaml V1.0.1. Basic40 is reused unchanged and compared exactly on every raw row during generation. No label enters feature computation.

- Stock histories stream across raw CSV chunks. Train and test history are concatenated per stock before strictly backward calculations, preserving EMA state and all observed-row lags. Absent dates are never inserted.
- Ordinary rolling statistics require a complete finite window. Correlations use centered direct windows with exactly w finite pairs and return NaN for zero variance. OLS uses chronological x=0..w-1; zero dependent variance gives NaN R².
- EMA follows pandas adjust=False, min_periods=span and default ignore_na=False. This is not a finite trailing-window SMA: historical EMA state survives internal gaps, while missing current close yields missing emadev.
- ATR propagates any missing required high/low/lag(close) component. Parkinson/GK reject nonpositive required prices; only GK's specified nonnegative variance clamp is used.
- Cross-sectional operations use finite values on the same date only. Empty finite amount aggregation gives NaN under the frozen missing-input rule. Market amount's full 20-date mean is computed on the observed date series across train/test.
- Limit event ages count observed rows, cap at 61, use 61 before any observed event, and preserve missing current flag as NaN. No price/flag filling or recoding.

Matrices are float64 Fortran-order files in outputs/features/full147; accepted Basic40 key arrays are reused without changes. Full feature quantiles/extreme keys and source fingerprints are persisted. Large matrices/models/predictions remain local; reports and hashes are versioned.

## E003 and reporting

E003 uses official lightgbm.LGBMRegressor with exactly the same configured parameters, target, train/validation keys, labels, flags and purge as E002. Only Full147 replaces Basic40. No validation fitting, early stopping or tuning is used.

Primary metrics come from the original organizer function, invoked on CSVs made from saved validation predictions in an isolated temporary directory. No hidden test labels or scores are involved. Original-source hash and default CSV parsing/tie behavior are preserved. Original E002 files are verified unchanged.

Official IC std ddof=1 and frozen auxiliary ddof=0 are separate fields. Auxiliary Top-Bottom spread uses equal floor-sized end groups and is explicitly not returned by the official script. Missing-label Top selection, repeated predictions, persistence and extra label-filtered diagnostic turnover are reported separately. None changes predictions or Official Score.

## Reproduce in PowerShell

```powershell
$env:PYTHONPATH='src'
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m stock_prediction.build_full147
.venv/Scripts/python.exe -m stock_prediction.verify_full147
.venv/Scripts/python.exe -m stock_prediction.e003
.venv/Scripts/python.exe -m stock_prediction.compare_e003
```

Generation/training refuses to overwrite completed feature/fold artifacts. The build is run once; existing outputs are verified, not regenerated during comparison. Individual fold command: `python -m stock_prediction.e003 --fold F1` after input verification.

Exact constant-valued decimal windows are identified from their input range, avoiding false nonzero variance caused by rounded means. The interrupted pre-correction attempt is recorded in outputs/e003_run_notes.md. Independent real-data cross-section/rank and full-date market-amount checks are saved in outputs/full147_verification.json.
