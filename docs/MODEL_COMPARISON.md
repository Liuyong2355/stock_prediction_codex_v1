# Frozen E006/E007 implementation and reproduction

Baseline: `24e979d361516af6d3d85de08e4d6b3e3be5b97b`.
Only E006/E007 are trained. No Phase C experiment is executed in this delivery.

## Controlled inputs

E006 uses `xgboost.XGBRegressor` and E007 uses `lightgbm.LGBMRanker`, instantiated
directly from the unchanged model catalog in `config/experiments_v1.yaml`.
XGBoost 3.4.1 was selected from the available official distribution before observing
any E006 result and pinned in `requirements-research.txt`. LightGBM stays at 4.6.0.
All other installed dependencies are identical to the E004/E005 environment.

The new entry point imports the existing `split_fold`, `materialize`, target,
official evaluator replay and missing-label diagnostics. No previous training,
feature, target, split or evaluation source is changed. Full147 input files are
verified against their accepted hashes. Feature inputs remain float64 and NaNs
are preserved. XGBoost's internal representation and default histogram/missing
handling belong to its official implementation; no attempt is made to reimplement
or adjust them. Its resolved `missing=np.nan` is logged explicitly as the string
`"NaN"` in JSON; this logging representation is never passed back to the model.

E006's supervised indices, original label values and rank-target vector hashes
must equal E004 before fit. The same 147 columns, original row order, batch size
(100,000), saved CSV precision, validation rows and evaluator are used. Only the
model and its frozen parameter package change. No validation set is passed to fit.

## Ranking group and relevance

1. Select the frozen fold, purge the final distinct candidate training date, and
   exclude nonfinite raw training labels using the existing split implementation.
2. Compute same-date average percentile ranks directly from those finite labels.
3. Construct integer relevance `min(floor(rank_pct * 10), 9)` exactly. Do not use
   `qcut`, break ties by stock code, or add 0.5 back to a rounded rank target.
4. Stable-sort the supervised slice by `trade_date`. Move X, y and row indices
   together. Original within-date stock order is retained.
5. Pass one group size per date to the official `LGBMRanker.fit(group=...)` API.
   Verify positive sizes, contiguous dates and `sum(group) == supervised rows`.

Training-only relevance, fit-order hashes, full group-date/count hashes, per-date
`ranking_groups.csv` and relevance histograms are persisted. Validation order is
unchanged and validation labels remain raw returns for official scoring. No group,
relevance, prediction filtering or postprocessing uses validation/test labels.

E007 vs E004 is a comparison of the frozen ranking experiment with rank regression.
The objective, integer relevance and required training order differ; it does not
isolate only the loss function. Library defaults, including ranking gains and
truncation, are not tuned or overridden. Native model serialization records the
actual library configuration.

## Reproduce (PowerShell, repository root)

```powershell
$env:PYTHONPATH='src'
$env:PYTHONUTF8='1'
.venv/Scripts/python.exe -m pip install -r requirements-research.txt
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m stock_prediction.e006_e007
.venv/Scripts/python.exe -m stock_prediction.compare_models
```

The recorded environment is `outputs/e006_e007_environment_lock.txt`; install that
lock for exact dependency reproduction. The runner requires that lock file to
match the E004/E005 lock plus only `xgboost==3.4.1`. Use
`--experiment E006 --fold F1` (or E007/F2/F3) to run an unfinished fold. Completed
folds are never overwritten. To repeat full training, use an isolated copy with
accepted inputs and E000–E005 artifacts/log, without E006/E007 fold outputs or log
entries. Saved-result verification runs `compare_models` only, without any fitting.

Fresh Git checkouts can normalize accepted CRLF text to LF. Run
`python -m stock_prediction.restore_model_text_contract` to check, and add
`--restore` only when needed. It reuses the prior SHA-256-constrained LF/CRLF
restorer, validates the entire set before writing, and rejects content changes.
It never changes accepted hashes, models, predictions or data. Previous-stage
results and source remain frozen byte-for-byte in the original workspace.

Large matrices, joblib models, XGBoost UBJSON and compressed predictions remain
local, under the established Git ignore policy. Metrics, group metadata, native
XGBoost configuration, source/artifact hashes, reports and tests are committed.
Existing manifests contain accepted local artifact paths; another environment
must make those artifacts available at their recorded paths. Git alone does not
contain all data required to retrain.

`compare_models` independently reconstructs new labels/order/groups using NumPy
date boundaries and SciPy ranks, reloads each new model and reproduces all validation
predictions, checks native serialization on a fixed sample, and invokes the original
organizer evaluator on all 24 E000–E007 saved prediction sets. E000's undefined
official IC/Score are preserved; no provisional result replaces them. Full repeated
training is tested on small synthetic panels, not claimed for a second full-data run.

See `outputs/e006_e007_comparison.md` and matching JSON/CSV, and
`outputs/e006_e007_validation.md` for executed checks and Phase C recommendations.
