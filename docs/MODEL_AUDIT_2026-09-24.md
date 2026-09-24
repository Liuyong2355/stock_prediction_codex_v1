# Model and strategy audit — 2026-09-24

## What is the main model?

The canonical primary **raw alpha model** is `main_alpha_xgb_rank162`, the
unchanged D0 F3 XGBoost artifact. It uses Full147 plus 15 frozen same-date
percentile-rank views, the continuous daily Rank target, and no prediction
postprocess. Its trained artifact is at `models/main_alpha_xgb_rank162/model.ubj`.
The rename is byte-for-byte: SHA-256
`096f50d7b306be28b22a1645c95d5e9e536478fb9423f0e1b3da8d48a84066ac`.
The F3 model was trained through 2023-12-28 after the one-day purge. It has
**not** been retrained using 2024 or the later test labels.

This naming describes the cleanest broad raw ordering signal. It does **not**
claim the largest historical Official Score among all strategies.

## Historical F1/F2/F3 comparison

F1, F2 and F3 are the 2022, 2023 and 2024 validation folds, not a separate
model family called “F”. Scores below use the frozen organizer evaluator.

| Candidate | Input/role | Mean Rank IC | Mean Top10 excess | Mean Score | Mean missing-label Top |
|---|---|---:|---:|---:|---:|
| E002 raw | Basic40 LightGBM | 0.066541 | 0.360977 | 0.291229 | 56.51% |
| E003 raw | Full147 LightGBM, raw target | 0.092061 | 0.468447 | 0.241129 | 27.51% |
| E004 raw | Full147 LightGBM, Rank target | 0.118149 | 0.443326 | 0.230187 | 4.62% |
| E005 raw | Full147 LightGBM, relative target | 0.096684 | 0.483008 | 0.261790 | 31.80% |
| E006 raw | Full147 XGBoost, Rank target | 0.118427 | 0.457543 | 0.233857 | 0.07% |
| **D0 raw / main_alpha_xgb_rank162** | **E006 plus 15 rank views** | **0.120835** | **0.477742** | **0.240126** | **0.02%** |
| D3 raw | D0 with Recent-2Y training | 0.104569 | 0.419696 | 0.216125 | 0.03% |
| D4 raw | D0 features with LightGBM | 0.120923 | 0.450875 | 0.236052 | 7.49% |
| E005+C0b | Frozen EMA/hysteresis, alpha=.30, exit=.25 | 0.075983 | 0.306100 | 0.371482 | 39.10% |
| E006+C0b | Same frozen postprocess | 0.098009 | 0.275437 | 0.358599 | 0.02% |
| C2-R1 postprocessed | Corrected LambdaRank | 0.096077 | 0.293574 | 0.399444 | 56.18% |

E002, E005 and C2 have higher historical Scores than D0, but a large fraction
of their official turnover Top consists of stocks without usable labels in
validation. Their scores cannot be read as clean evidence of stronger
tradable alpha. E006+C0b is the strongest documented **low-missing** historical
scoring strategy, but it is based on the older 147-feature model and gives up
some IC and Top10 excess to reduce turnover. The C0b parameter search is closed.
No untested D0+C0b combination is promoted here.

## One later test period, frozen artifacts only

The 2025-01-02 to 2026-06-05 local labels were reconstructed retrospectively
from the next test row's close. Each prediction and frozen postprocess was
written before test labels were loaded for scoring. This was a single model
comparison, not a new validation fold or a parameter search.

| Candidate | Rank IC | Top10 excess | Official turnover | Score | Diagnostic turnover | Missing-label Top |
|---|---:|---:|---:|---:|---:|---:|
| **D0 raw** | **0.129418** | 0.461515 | 0.840380 | 0.238108 | 0.843268 | 0.03% |
| D3 raw | 0.110341 | 0.379274 | 0.831759 | 0.208391 | 0.834652 | 0.03% |
| D4 raw | 0.127397 | 0.436533 | 0.832653 | 0.232123 | 0.835594 | 0.11% |
| E005 raw | 0.096470 | **0.498468** | 0.867473 | 0.227886 | 0.868840 | 2.70% |
| E006 raw | 0.125750 | 0.431406 | 0.830918 | 0.230446 | 0.833977 | 0.03% |
| E005+C0b | 0.085484 | 0.345900 | 0.228879 | **0.369300** | 0.254768 | 0.63% |
| E006+C0b | 0.108067 | 0.254035 | 0.176712 | 0.366424 | 0.203640 | **0.03%** |

The E005+C0b test Score exceeds E006+C0b by only `0.002876`, while E005+C0b
had severe validation missing-label occupancy. The later period alone does not
erase that historical failure mode. D0 remains the primary raw alpha model;
E006+C0b remains a documented high-score, low-missing historical strategy,
not the renamed main model. F1/F2/F3 and the later test are different years;
their scores should not be combined into a new mean for model selection.

## Research routes and cleanup boundary

D2-A and D2-B failed target experiments. D3 is an auxiliary candidate, D4 was
not promoted, and D5's initially large positive `rank(y)-rank(D0)` IC was
partly mechanical. The corrected D5 conditional IC is small and negative;
no residual model is promoted. Their result reports remain as historical
evidence, indexed by `archive/completed_research_2026-09-24/README.md`.

Only proven scratch artifacts from an interrupted E003 run and an obsolete
pre-correction Full147 build are eligible for physical deletion. **Deletion is
pending:** the execution policy rejected the cleanup command. The exact
verified targets are:

- `outputs/features/full147_before_zero_variance_fix/train_full147.npy`
  (9,290,811,728 bytes);
- `outputs/features/full147_before_zero_variance_fix/test_full147.npy`
  (1,881,129,728 bytes);
- `outputs/baselines/cache/e003_interrupted_zero_variance_fix/F1/training_work.npy`
  (4,197,000,656 bytes).

Together they occupy about 14.3 GiB. Current raw data, current
Full147/Basic40 matrices, labels, frozen model artifacts, validation reports
and local test submissions are retained. The test X/Y files and their transfer
archive stay local and are not pushed to Git.

Sources: `outputs/e006_e007_comparison.md`, `outputs/phase_c0b/phase_c0b_turnover_comparison.md`,
`outputs/phase_c2/phase_c2_comparison.md`, `outputs/phase_d0_e006_rank_view/comparison.md`,
`outputs/phase_d3_recent_2y/comparison.md`, `outputs/phase_d4_lightgbm_rank_regression/comparison.md`,
`outputs/frozen_test_2025_2026/comparison.md`, and
`outputs/frozen_test_2025_2026/legacy_strategies.md`.
