# Frozen model test period, first look

This is a retrospective local evaluation of labels reconstructed from the next test row's close. No model was retrained or selected with these labels.

F3 models were frozen before this evaluation and trained through 2023 at the latest; 2024 was their validation year.

| Model | Rank IC | Top10 annual excess | Official turnover | Official Score | Diagnostic turnover | Missing-label Top fraction |
|---|---:|---:|---:|---:|---:|---:|
| D0 | 0.129418 | 0.461515 | 0.840380 | 0.238108 | 0.843268 | 0.03% |
| D3 | 0.110341 | 0.379274 | 0.831759 | 0.208391 | 0.834652 | 0.03% |
| D4 | 0.127397 | 0.436533 | 0.832653 | 0.232123 | 0.835594 | 0.11% |

The table is a single test-period observation, not an extra training fold or a basis for tuning.
