# Initialization and audit validation

- `python -m pytest -q`: 15 tests passed.
- `python -m pip check`: no broken requirements.
- Optional Phase A model dependencies resolve on Windows / Python 3.14 (dry run only; not installed or trained).
- Tests cover raw preservation, leading-zero stock identifiers, NaN/inf, duplicates across chunks, market gaps versus missing prices, stock-isolated label comparison, no skipping missing closes, train/test boundary diagnostic scope, schema/numeric rejection, full-file encoding validation, report output and frozen feature-set counts.
- Both real CSVs were fully scanned: 7,900,350 training rows and 1,599,600 test rows.
- SHA-256 before raw-file relocation, after relocation, and during full audit agree for both files.
- The audit implementation SHA-256 matches the value embedded in data_audit_summary.json. Config SHA-256 values are recorded there too.
- No E000–E007, features, supervised model, evaluator or competition result was implemented or executed.

The raw files have been archived unchanged into data/raw/. They and the local virtual environment are excluded from Git. Full per-column and per-stock diagnostics are in data_audit_summary.json; data_audit_report.md distinguishes confirmed facts, warnings, unresolved questions and blockers.

## Cross-field flag audit extension

- Full scan detects 4 train rows and 0 test rows with both flags equal to 1. Counts and up to 10 complete raw-column examples are recorded in the report and JSON.
- Single-column 0/1 validation remains separate. These observations are warning / unresolved observation only, never data errors or blockers.
- Additional tests cover both train/test schemas, sample cap across chunks, independent flag domains, unchanged raw values/bytes, and warning-only report classification.
- Full rerun preserved both raw SHA-256 values and data_version.
