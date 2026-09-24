# Next research handoff

Repository: `https://github.com/Liuyong2355/stock_prediction_codex_v1`

Formal baseline before Phase C/D: `4c0aa59898156a6c53721e4d1645ee1a766417af`.

## Read in this order

1. `AGENTS.md`
2. `README.md`
3. `docs/DECISIONS.md`, especially D021–D026
4. `outputs/e006_e007_comparison.md`
5. `outputs/phase_c2/phase_c2_comparison.md`
6. `outputs/phase_d0_e006_rank_view/comparison.md`
7. `outputs/phase_d1_lambdarank_rank_view/comparison.md`
8. `config/phase_d0_e006_rank_view.yaml`

## Established conclusions

- Full147 contains materially more predictive information than Basic40, although
  raw Official Score can be lower because turnover rises.
- E006 Full147 + XGBoost + Rank is the clean alpha baseline: positive and stable
  Rank IC with negligible missing-label Top occupancy.
- Historical E007 is retired. Corrected LambdaRank restores positive signal, but
  its low official turnover is strongly missing-label-sensitive.
- Turnover-grid expansion and E005/E006 fusion are closed research routes.
- New RSI/MACD/ATR/Bollinger features have not shown model-level independent
  value. Full147 already contains ATR; the teammate ATR augmentation did not beat
  its base ranker on mean Score.

## Phase D single-variable result

The teammate's 36 core factors were mapped to 30 comparable Full147 sources.
Fifteen already had `csr_*` features, so only 15 new same-date percentile ranks
were added while retaining all original Full147 columns.

### D0 — E006

- Rank IC: `0.118427 -> 0.120835`, delta `+0.002408`, improved 3/3 folds.
- Top10% excess: `0.457543 -> 0.477742`, delta `+0.020199`, improved 3/3.
- Official Score: `0.233857 -> 0.240126`, delta `+0.006269`, improved 3/3.
- Mean missing-label Top fraction: `0.02%`.

This is a clean structural representation gain.

### D1 — corrected LambdaRank raw predictions

- Rank IC: `0.113677 -> 0.115950`, delta `+0.002273`, improved 3/3 folds.
- Top10% excess: mean delta `+0.003242`, improved only 1/3.
- Official Score: mean delta `+0.003845`, improved only 1/3.

The feature view generalizes as an ordering signal, but not as a stable
LambdaRank tail-return or final-score improvement.

### D2-A — Top10-stretched continuous rank target

- Rank IC: `0.120835 -> 0.080186`, delta `-0.040649`, improved 0/3 folds.
- Top10% excess: `0.477742 -> 0.398918`, delta `-0.078824`, improved 0/3.
- Official Score: `0.240126 -> 0.201676`, delta `-0.038450`, improved 0/3.
- Mean official turnover improves by `0.004856`, but contributes only `+0.001457`
  to Score; mean missing-label Top fraction rises to `0.72%`.

The fixed coefficient-10 stretch destroys both broad ordering and tail return,
especially in F2. Stop this target route; do not tune its threshold/coefficient
or append another target formula.

### D2-B — Top10 binary target

- Rank IC: `0.120835 -> -0.066144`, delta `-0.186979`, improved 0/3 folds.
- Top10% excess: `0.477742 -> -0.403016`, delta `-0.880758`, improved 0/3.
- Official Score: `0.240126 -> -0.027615`, delta `-0.267741`, improved 0/3.
- Mean daily prediction Spearman versus D0: `-0.390155`.
- Mean Top10 Jaccard / overlap versus D0: `0.043645 / 0.079474`.
- Mean missing-label Top fraction rises from `0.02%` to `15.65%`.

D2-B is materially different from D0 but consistently anti-predictive, not an
independent tail alpha. Formally stop the entire dedicated Tail-target route;
do not tune or fuse this classifier.

### D3 — Recent-2Y training window

- Rank IC: `0.120835 -> 0.104569`, delta `-0.016266`, improved 0/3 folds.
- Top10% excess: `0.477742 -> 0.419696`, delta `-0.058046`, improved 0/3.
- Official Score: `0.240126 -> 0.216125`, delta `-0.024001`, improved 0/3.
- Mean daily prediction Spearman versus D0: `0.786118`.
- Mean Top10 Jaccard / overlap versus D0: `0.334588 / 0.489518`.
- Mean missing-label Top fraction remains negligible at `0.03%`.

Recent-2Y is not a stronger main model, but it retains positive IC, excess, and
Score in all folds with materially different Top10 selections. Preserve it as a
second Alpha candidate only. Stop window-length/time-decay research and do not
infer permission to fuse from this handoff.

### D4 — LightGBM Rank Regression

- Rank IC: `0.120835 -> 0.120923`, delta `+0.000088`, improved 1/3 folds.
- Top10% excess: `0.477742 -> 0.450875`, delta `-0.026867`, improved 1/3.
- Official Score: `0.240126 -> 0.236052`, delta `-0.004074`, improved 1/3.
- Prediction Spearman versus D0: `0.787081`.
- Top10 Jaccard / overlap: `0.552922 / 0.701200`.
- Per-model exclusive Top10 fraction: `29.88%`.
- Missing-label Top fraction: mean `7.49%`, F2 `19.53%`.

D4 has stable positive standalone alpha but fails the clean-second-model rule:
its selection/turnover difference is missing-label-sensitive and Score improves
only in F2. Stop plain LightGBM regression; do not tune or fuse it.

## Current stopping boundary

Do not reopen turnover tuning, fusion grids, C2-R2 full replacement, or broad
technical-indicator expansion. Also do not continue the D2-A continuous target
stretch route or any dedicated Tail-target classifier following D2-B. Do not
test other training windows or time-decay variants after D3. Do not propose many
small experiments. Do not reopen plain LightGBM regression after D4.

## Question for the next researcher

Given the evidence above, identify one structural next step most likely to
produce a substantially larger Score gain than the observed `+0.0063` E006
increment. Distinguish improvements to ordering alpha from improvements to the
Top10% tail. Recommend at most two single-variable experiments, and explicitly
state which current routes should remain stopped.

Large local matrices, trained models and OOF prediction files are intentionally
Git-ignored. The committed JSON/Markdown results contain the metrics needed for
research decisions.
# Current handoff after D5 (2026-09-23)

D0 remains the formal main model. D3 remains only an auxiliary second-alpha
candidate; D4 is rejected because of abnormal missing-label Top occupancy.

D5 completed a diagnostic-only residual-alpha audit using frozen D0 validation
predictions; it trained no model and performed no fusion. Stable residual signal
was found primarily in the full cross-section and D0 middle60, led by volatility
and then volume-price features. The signal is weak/inconsistent in D0 Top20 and
must not be described as tail alpha. See
`outputs/phase_d5_residual_alpha/comparison.md` and D031.

Correction on 2026-09-24: the large D5 positive IC against `rank(y)-rank(D0)`
is partly mechanical. The read-only direct/partial-IC audit finds only small,
consistently negative conditional true-return signal in volatility and
volume-price; see `outputs/phase_d5_residual_alpha/partial_audit.md` and revised
D031. Do not train a residual model on in-sample D0 predictions. A future model
experiment first needs a frozen, leakage-safe training and validation design.

Frozen D0/D3/D4 F3 models were scored once on the retrospectively reconstructed
2025-01-02 to 2026-06-05 test labels. Scores are `0.238108/0.208391/0.232123`;
D0 stays strongest. Treat these as one-time test observations, not tuning data.
See `outputs/frozen_test_2025_2026/comparison.md` and D032. Do not reopen generic
model-zoo, tail-target, training-window, turnover, or LightGBM tuning routes.
