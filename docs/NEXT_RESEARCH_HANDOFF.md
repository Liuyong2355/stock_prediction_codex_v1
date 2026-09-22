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

## Current stopping boundary

Do not reopen turnover tuning, fusion grids, C2-R2 full replacement, or broad
technical-indicator expansion. Do not propose many small experiments.

## Question for the next researcher

Given the evidence above, identify one structural next step most likely to
produce a substantially larger Score gain than the observed `+0.0063` E006
increment. Distinguish improvements to ordering alpha from improvements to the
Top10% tail. Recommend at most two single-variable experiments, and explicitly
state which current routes should remain stopped.

Large local matrices, trained models and OOF prediction files are intentionally
Git-ignored. The committed JSON/Markdown results contain the metrics needed for
research decisions.
