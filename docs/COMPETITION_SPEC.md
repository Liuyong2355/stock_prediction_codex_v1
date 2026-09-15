# Competition Specification

**Status:** V1 transcription of the organizer task PDF.  
**Primary source:** `reference/赛题五-更新.pdf`

## 1. Task

For every `(ts_code, trade_date)` row in the test set, predict `y_ret_1d`, the next-trading-day return:

```text
y_ret_1d(t) = close(t+1) / close(t) - 1
```

The competition uses full-market A-share daily OHLCV-style data.

## 2. Files

- `训练集.csv`: features X plus label Y.
- `测试集_X.csv`: features X only.
- Required submission: `submission.csv`.

## 3. Raw fields

| Field | Type | Meaning |
|---|---|---|
| `ts_code` | str | Stock code, e.g. `000001.SZ`, `600000.SH` |
| `trade_date` | int | Trading date, e.g. `20180102` |
| `open` | float | Back-adjusted open price |
| `high` | float | Back-adjusted high price |
| `low` | float | Back-adjusted low price |
| `close` | float | Back-adjusted close price |
| `vol` | float | Trading volume in shares |
| `amount` | float | Trading amount in yuan |
| `flag_limit_up` | int | 1 if limit-up on that date, else 0 |
| `flag_limit_down` | int | 1 if limit-down on that date, else 0 |
| `y_ret_1d` | float | Next-day return; training set only |

Prices are already back-adjusted so corporate-action discontinuities have been removed for indicator computation.

## 4. Dataset scale

| Item | Training | Test |
|---|---:|---:|
| Stocks | 4,650 | 4,650 |
| Rows | 7,900,350 | 1,599,600 |
| Date range | 2018-01-02 to 2024-12-31 | 2025-01-02 to 2026-06-08 |

Approximately 14.5% of training `y_ret_1d` values are NaN. The organizer attributes these mainly to suspension/not-yet-listed situations. The participant must handle them.

## 5. Submission

`submission.csv` must contain exactly:

```text
ts_code,trade_date,pred
000001.SZ,20250102,0.0052
000001.SZ,20250103,-0.0031
...
```

`ts_code` and `trade_date` must correspond to the test rows. `pred` is the predicted next-day return/score used by the evaluator.

## 6. Official metrics

### 6.1 Mean daily Rank IC — weight 40%

For each test trading date:

1. Take all `pred` and `y_ret_1d`.
2. Exclude rows whose `y_ret_1d` is NaN.
3. Compute Spearman rank correlation:

```text
IC_t = spearman(pred_t, y_ret_1d_t)
```

Final:

```text
IC_mean = mean(IC_t)
```

Higher is better.

### 6.2 Top-group annualized excess return — weight 30%

For each date:

1. Exclude `flag_limit_up == 1`.
2. Exclude rows whose `y_ret_1d` is NaN.
3. Sort remaining stocks by `pred` descending.
4. Split into 10 groups; Top1 is the highest-prediction group (top ~10%).
5. Compute Top1 mean realized return.
6. Compute mean realized return of all remaining eligible stocks.
7. Daily excess = Top1 mean return - eligible-market mean return.

Final:

```text
annualized_excess = mean(daily_excess) * 252
```

Higher is better.

### 6.3 Prediction turnover — weight 30%

For each date, starting from the second date:

1. Exclude `flag_limit_up == 1`.
2. Sort remaining stocks by `pred` descending.
3. Let `Set_t` be the Top1 group / top ~10% stock set.
4. Jaccard-distance turnover:

```text
turnover_t = 1 - |Set_t ∩ Set_(t-1)| / |Set_t ∪ Set_(t-1)|
```

Final:

```text
mean_turnover = mean(turnover_t)
```

Lower is better.

**Important:** the PDF's turnover rule does not explicitly say to exclude rows with NaN `y_ret_1d`; do not add that filter unless the organizer `evaluate.py` does.

### 6.4 Final score

```text
final_score =
    0.40 * IC_mean
  + 0.30 * annualized_excess
  + 0.30 * (1 - mean_turnover)
```

The organizer PDF gives typical ranges, but no separate normalization step is described.

## 7. Auxiliary metrics listed by the organizer

Not part of the final score:

- `ICIR = IC mean / IC standard deviation`
- fraction of trading days with Rank IC > 0
- Top1 annualized absolute return
- Top1 - Bottom1 annualized long-short return spread

## 8. Competition constraints

- Feature engineering may use only the fields in the training/test data.
- Future information is prohibited.
- Test Y is hidden during the competition and used only for final scoring.
- Open-source tools and model frameworks are allowed.
- Training-label NaNs must be handled by the participant.

## 9. Evaluator edge cases that are not fully specified by the PDF

Until the organizer-provided `evaluate.py` is available, the following must be treated as unresolved implementation details rather than guessed as "official":

- exact group-size behavior when eligible stock count is not divisible by 10;
- tie handling at the Top1 boundary;
- behavior for constant predictions / undefined Spearman values;
- behavior for missing or infinite predictions;
- exact handling of dates with too few valid rows.

The organizer `evaluate.py` overrides any provisional implementation on these points.

## 10. Local implementation status

The organizer evaluate.py remains absent. All internally implemented evaluation is provisional, not official. Auxiliary IC standard deviation is frozen at ddof=0 in config/experiments_v1.yaml, subject to organizer metric precedence.
