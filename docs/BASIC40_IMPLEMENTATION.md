# Basic40 implementation and audit

The user's current stage name is B1. Its scope is the Basic40 step of the existing Phase A baseline protocol, not the protocol's Full147 Phase B. The frozen feature specification and YAML are unchanged at V1.0.1.

## Computation

- `compute_basic40` selects only identifiers and eight source columns, sorts by stock/date and computes each stock separately. Labels and other columns are ignored; duplicate/missing keys fail explicitly.
- Every lag counts observed rows, including rows with missing prices. No calendar expansion or filling is performed. Return features use the specified two endpoints; intermediate missing observations do not invalidate endpoint-only returns.
- Rolling means require exactly w finite inputs, including today. Return standard deviation requires w valid one-row returns, therefore w+1 raw closes; ddof=0. Infinite inputs in the working copy and infinite derived results become NaN. No raw file is modified.
- Epsilon is added only where the frozen formula specifies it. No winsorization, normalization, clipping, label-based row filtering or new trading eligibility filter is applied.
- Same-day flags are preserved, including both flags equal to 1. They remain unresolved observations rather than data errors.
- Basic40 contains no EMA, rolling correlation, OLS, market aggregates or cross-sectional ranks. Their frozen rules remain in the contract; their implementations are outside this task. The cross-sectional test checks Basic40's independence from other stocks on the same date.

## Full-panel generation and storage

The streaming builder requires the accepted raw audit's stock/date ordering, equal train/test stock sets, non-overlapping periods and matching raw-file hashes. A stock's full train/test history is computed before splitting outputs; test-date rolling history includes train-date observations. Input CSV chunk boundaries never reset a stock's history.

Each split has a float64 NumPy matrix with exactly 40 feature columns, stored in column-major order for bounded-memory column audits, and an aligned structured NumPy key array. Feature row i corresponds to key row i. The matrix excludes identifiers, raw OHLC and labels. Partial files are not accepted as completed artifacts; the manifest is written after generation and auditing finish. Completed output files are not silently overwritten.

```python
import json
import numpy as np

with open('outputs/basic40_manifest.json', encoding='utf-8') as stream:
    manifest = json.load(stream)
split = manifest['datasets']['train']['files']
X = np.load(split['matrix']['path'], mmap_mode='r', allow_pickle=False)
keys = np.load(split['keys']['path'], mmap_mode='r', allow_pickle=False)
names = manifest['feature_names']
# X[i, j] belongs to keys[i], feature names[j].
```

Artifacts remain local under `outputs/features/basic40/` (excluded from Git); the tracked manifest records absolute paths, row counts, column order, sizes and SHA-256. Raw data/version and feature contract hashes are recorded in the audit JSON. The recorded base Git commit identifies the pre-build repository; implementation file hashes identify the exact feature source, including changes committed after the build.

## Validation and audit interpretation

Unit tests compare all 40 columns against direct NumPy window calculations, test exact first-valid positions, ddof=0, missing and infinite inputs, absent market-date rows, stock isolation, same-date independence, future perturbation, ignored labels, test warm-up, unchanged raw flags, chunk boundaries and persisted output alignment. EMA/Full147 are not implemented solely to add irrelevant tests.

Full-data audit reports per-split finite/NaN counts and ratios, inf counts, exact finite-only min/p01/p25/p50/p75/p99/max, and min/max stock-date examples. Selected review examples include raw history and independent scalar checks. These are descriptive statistics only and must not be used as globally fitted training preprocessors. No competition scores or model experiments are calculated.
