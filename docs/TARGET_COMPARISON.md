# E004 / E005 frozen target comparison

Baseline: `eb9660a`. This stage runs E004 (Full147 + LightGBM + rank) and E005
(Full147 + LightGBM + relative). Existing E003 raw-target artifacts are the control.

## Training and evaluation boundary

`e003.run_fold` is shared by all three experiments. The materialized float64 feature
matrix, column order, observation order, fold splitting, purge, finite-label filter,
LightGBM settings, 800 iterations, prediction batches and CSV roundtrip are unchanged.
No validation set is passed to fit. The runner accepts only E003/E004/E005; the new
stage entry point accepts only E004/E005.

After splitting and purging, `training_target` receives only supervised training
dates and finite raw labels. Rank uses average same-date percentile ranks minus 0.5.
Relative subtracts the same-date median. All training stocks participate, including
limit-up stocks when their labels are finite, exactly as in E003. No ranking-group
sort, label scaling, clipping or sample reweighting is introduced. Nonfinite values
are rejected by the target constructor because filtering belongs to `split_fold`.

Validation labels always remain raw returns. Predictions are saved for every
validation row, including missing labels. The unmodified organizer evaluator is
called on those saved predictions. Its IC standard deviation uses ddof=1; the
frozen auxiliary ddof=0 remains separate. Diagnostic turnover reselects daily Top
after excluding missing raw labels and never affects fit, predictions or Score.

## Provenance and verification

The input verifier checks accepted raw data, frozen Full147 matrices/keys and label
cache hashes. It snapshots E000–E003 artifacts, existing reports and the original
experiment log. Each fold checks E003 split metadata, resolved/configured parameters,
feature manifest, frozen config hashes and Python/NumPy/pandas/LightGBM versions.
New results record ordered train/validation index hashes, raw/transformed training
label hashes, source hashes, official script hash and artifact hashes.

`compare_targets` independently constructs every new training label using NumPy
date grouping and SciPy ranks, verifies every artifact, reproduces every validation
prediction from the saved model, checks native text serialization on a fixed sample,
and invokes the original evaluator again for all nine experiment-fold combinations.
It checks strict E003 controls and the preserved original log plus exactly six new
entries. Per-fold and cross-fold Score contributions are checked to add up.

The inherited split metadata contains the historical string “Full causal Basic40
reused”. It is preserved for exact comparison; all three experiments actually use
the frozen Full147 matrix identified by its manifest hash and 147 saved model names.

## Reproduce in PowerShell

From the repository root, with the existing environment and accepted local data:

```powershell
$env:PYTHONPATH='src'
$env:PYTHONUTF8='1'
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m stock_prediction.e004_e005
.venv/Scripts/python.exe -m stock_prediction.compare_targets
```

The training command refuses to overwrite a completed fold. For a not-yet-completed
individual fold, use `--experiment E004 --fold F1` (or E005/F2/F3). Every invocation
verifies inputs. For completed results, run only `compare_targets`; it does not fit
models or overwrite predictions. To repeat full training, use a separate checkout
with accepted input artifacts and E000–E003 outputs, without E004/E005 outputs or
their six experiment-log entries. Keep the recorded library versions and runtime.

The pre-existing baseline has CRLF working-tree files whose Git blobs are LF
(including frozen YAML and `baselines.py`). Its accepted hashes refer to the original
working-tree bytes. In a fresh checkout, first run
`python -m stock_prediction.restore_text_contract` to check, then add `--restore`
if needed. The helper considers only LF/CRLF variants and writes only bytes matching
the existing accepted SHA-256; changed content fails before any write. Large data,
models, predictions and accepted hashes are never repaired or modified by this helper.

Large matrices, prediction files and model files remain local under the established
repository policy; compact results, SHA-256 provenance, daily metrics and reports
are committed. A fresh checkout therefore also needs those local accepted artifacts
or reproduction of prior stages. Repeated frozen fits are tested on a small synthetic
panel; full-size repeat training is not claimed. Full-size saved-model prediction
reproduction is checked for all nine folds.

Results: `outputs/e003_e004_e005_comparison.md` and matching JSON/CSV.
Tests and executed validation: `outputs/e004_e005_validation.md`.
