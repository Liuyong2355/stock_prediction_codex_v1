# Frozen E-series strategy check on the 2025-2026 test period

The C0b parameters were fixed in F1/F2 before this period: alpha=0.30 and exit_fraction=0.25. The F3 model files were not retrained. These results are a one-time comparison and do not reopen the turnover grid.

| Strategy | Rank IC | Top10 excess | Official turnover | Score | Diagnostic turnover | Missing-label Top |
|---|---:|---:|---:|---:|---:|---:|
| D0 raw | 0.129418 | 0.461515 | 0.840380 | 0.238108 | 0.843268 | 0.03% |
| E005 raw | 0.096470 | 0.498468 | 0.867473 | 0.227886 | 0.868840 | 2.70% |
| E005 C0b | 0.085484 | 0.345900 | 0.228879 | 0.369300 | 0.254768 | 0.63% |
| E006 raw | 0.125750 | 0.431406 | 0.830918 | 0.230446 | 0.833977 | 0.03% |
| E006 C0b | 0.108067 | 0.254035 | 0.176712 | 0.366424 | 0.203640 | 0.03% |

The local labels are reconstructed retrospectively from the next test row's close.
