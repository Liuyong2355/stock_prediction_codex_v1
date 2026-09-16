# Stock prediction V1

Read AGENTS.md and docs/index.md first. Completed stages include audits, Basic40,
Full147 and E000–E007. The latest stage runs only frozen E006/E007, with direct
official XGBoost/LightGBM training, organizer scoring and preserved missing-label diagnostics.

- Output index: [artifacts organized by research phase](outputs/README.md).
- Current results: [Phase C0b bounded turnover optimization](outputs/phase_c0b/phase_c0b_turnover_comparison.md).
- Prior results: [E000–E007 comparison and Phase C candidates](outputs/e006_e007_comparison.md).
- Validation: [Phase C0b checks](outputs/phase_c0b/phase_c0b_turnover_validation.md).
- Reproduction: [model comparison commands and frozen-byte handling](docs/MODEL_COMPARISON.md).
- Current model environment: `outputs/e006_e007_environment_lock.txt`; new model pin in `requirements-research.txt`.
- Prior target results: [E003/E004/E005 comparison](outputs/e003_e004_e005_comparison.md).

Phase C, tuning, feature ablations, ensembles and turnover postprocessing have not been run.

## Initial audit setup — historical Windows / PowerShell instructions

```powershell
py -3.14 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = 'src'
.venv/Scripts/python.exe -m stock_prediction.audit --root .
.venv/Scripts/python.exe -m pytest
```

To reproduce the exact audited environment including transitive dependencies, install from `outputs/environment-lock.txt` instead of `requirements.txt`. The optional model dependency pins were resolved with a dry run on Python 3.14; no model packages were installed and runtime model compatibility has not yet been tested.

`requirements-models.txt` pins the additional Phase A model dependencies; they are not installed or used at the audit stage. XGBoost/Full147 are outside this stage.

Raw files live in `data/raw/` and are excluded from Git. The audit reads every row, preserves all raw values, and writes `outputs/data_audit_report.md` and `outputs/data_audit_summary.json`. `outputs/data_manifest.json` records raw-file provenance. `outputs/environment-lock.txt` records the installed environment. No missing trading dates or values are filled.

The reader uses chunks; only identifiers, close, label and missingness masks are retained for temporal checks. The observed union of dates is an empirical market calendar, not an externally verified exchange calendar. No labels are reconstructed for training.

## Basic40 generation (user stage B1)

```powershell
$env:PYTHONPATH = 'src'
.venv/Scripts/python.exe -m stock_prediction.build_basic40 --root .
```

This reads the accepted raw-data audit and verifies its data fingerprints. Complete stock histories span CSV chunks and the train/test boundary. Generated matrices and aligned keys are in `outputs/features/basic40/`; column order, shapes and SHA-256 values are in `outputs/basic40_manifest.json`. The builder refuses to overwrite completed artifacts. Generation uses only NumPy/pandas already installed, and no labels, fits, clipping or calendar filling.

Read `docs/BASIC40_IMPLEMENTATION.md` for implementation details and loading instructions. Descriptive statistics and manual-check examples are in `outputs/basic40_audit_report.md` and `outputs/basic40_audit_summary.json`.

## Frozen B2 baselines

```powershell
.venv/Scripts/python.exe -m pip install -r requirements-models.txt
.venv/Scripts/python.exe -m pytest -q
$env:PYTHONPATH = 'src'
.venv/Scripts/python.exe -m stock_prediction.baselines --root .
```

Run only after the baseline tests and official-library smoke tests pass. The command verifies training input artifacts and runs E000, E001, E002 sequentially on F1–F3. It refuses to overwrite completed folds. `--experiment E001 --fold F2` selects one unfinished fold after input verification. Scores are provisional; read `docs/PROVISIONAL_EVALUATOR.md` for exact edge assumptions. Frozen config files remain unchanged. Predictions/model binaries are stored locally under outputs/baselines, with hashes in tracked per-fold result.json files. Summary and experiment logs are tracked.
