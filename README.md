# Stock prediction V1

Read AGENTS.md and docs/index.md first. Current scope: initialization and full raw-data audit only.

## Windows / PowerShell

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
