# Feature Specification V1.0.1

**Canonical machine-readable contract:** `../config/features_v1.yaml`

## 1. Purpose

Feature Set V1 turns the organizer-provided OHLC/volume/amount/limit fields into a causal stock-state representation for next-day cross-sectional return prediction.

V1 has two frozen feature sets:

- **Basic40** — the first implementation and baseline set.
- **Full147** — the first full research set.

The exact membership is canonical in `config/features_v1.yaml`.

## 2. Global implementation invariants

1. Sort by `ts_code`, then ascending `trade_date`.
2. Every per-stock lag/rolling/EMA/regression feature is computed independently inside `ts_code`.
3. Every window is backward-looking and includes the current date unless a formula explicitly says otherwise.
4. No feature may use `t+1` or later data when producing a feature for date `t`.
5. Same-day cross-sectional operations use only rows from that same `trade_date`.
6. Use `epsilon = 1e-12` where the formula specifies an epsilon denominator.
7. Rolling standard deviations use `ddof=0`.
8. Replace `+inf` and `-inf` with NaN.
9. Do not globally fill feature NaNs with zero.
10. `ts_code` is an identifier in V1, not a model feature.
11. Raw absolute `open/high/low/close` are source values, not V1 model features.
12. Raw `vol` and `amount` are source values; V1 models use the derived features defined here.
13. Feature computation should occur on the chronological raw panel before fold slicing, provided every transformation is strictly causal. This preserves legitimate history for validation dates.

## 3. Feature counts

- Return/momentum/reversal: 10
- K-bar / price shape: 12
- Trend / price position: 31
- Volatility: 20
- Volume / amount / volume-price: 26
- Liquidity: 4
- Limit-up / limit-down: 10
- Market state: 10
- Relative-to-market: 8
- Cross-sectional ranks: 16

Total:

```text
10 + 12 + 31 + 20 + 26 + 4 + 10 + 10 + 8 + 16 = 147
```

## 4. Basic40 membership

Basic40 is deliberately small and must be implemented before Full147.

```text
ret_1
ret_2
ret_3
ret_5
ret_10
ret_20
ret_30
ret_60
kmid
klen
kmid2
kup
kup2
klow
klow2
ksft
ksft2
madev_3
madev_5
madev_10
madev_20
madev_30
madev_60
stdret_5
stdret_10
stdret_20
stdret_60
volratio_5
volratio_10
volratio_20
volratio_60
amtratio_5
amtratio_10
amtratio_20
amtratio_60
gap_open
gap_high
gap_low
flag_limit_up
flag_limit_down
```

## 5. Formula conventions

- `lag(x,k)`: value of x k trading observations earlier within the same `ts_code`.
- `rolling_mean(x,w)`: mean over the trailing w observations including t.
- `rolling_std(x,w,ddof=0)`: population standard deviation over the trailing w observations.
- `rolling_corr(a,b,w)`: Pearson correlation over the trailing w observations.
- `ema(..., adjust=False, min_periods=w)`: causal pandas-style exponentially weighted mean.
- `rank_pct(x within trade_date, method='average')`: percentile rank among finite same-date values using average ties.
- OLS trend features require exactly w finite observations in chronological order.

## 6. Feature definitions

## A. Return / momentum / reversal

### `ret_1`

- **Purpose:** 1-trading-day close-to-close return.
- **Formula:** `close_t / lag(close,1) - 1`
- **Raw inputs:** `close`
- **Window / history:** 1
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `ret_2`

- **Purpose:** 2-trading-day close-to-close return.
- **Formula:** `close_t / lag(close,2) - 1`
- **Raw inputs:** `close`
- **Window / history:** 2
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `ret_3`

- **Purpose:** 3-trading-day close-to-close return.
- **Formula:** `close_t / lag(close,3) - 1`
- **Raw inputs:** `close`
- **Window / history:** 3
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `ret_5`

- **Purpose:** 5-trading-day close-to-close return.
- **Formula:** `close_t / lag(close,5) - 1`
- **Raw inputs:** `close`
- **Window / history:** 5
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `ret_10`

- **Purpose:** 10-trading-day close-to-close return.
- **Formula:** `close_t / lag(close,10) - 1`
- **Raw inputs:** `close`
- **Window / history:** 10
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `ret_20`

- **Purpose:** 20-trading-day close-to-close return.
- **Formula:** `close_t / lag(close,20) - 1`
- **Raw inputs:** `close`
- **Window / history:** 20
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `ret_30`

- **Purpose:** 30-trading-day close-to-close return.
- **Formula:** `close_t / lag(close,30) - 1`
- **Raw inputs:** `close`
- **Window / history:** 30
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `ret_60`

- **Purpose:** 60-trading-day close-to-close return.
- **Formula:** `close_t / lag(close,60) - 1`
- **Raw inputs:** `close`
- **Window / history:** 60
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `ret_5_minus_20`

- **Purpose:** Short-vs-medium return spread.
- **Formula:** `ret_5 - ret_20`
- **Depends on:** `ret_5`, `ret_20`
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `ret_20_minus_60`

- **Purpose:** Medium-vs-long return spread.
- **Formula:** `ret_20 - ret_60`
- **Depends on:** `ret_20`, `ret_60`
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

## B. Same-day K-bar / price shape

### `kmid`

- **Purpose:** Signed open-to-close body scaled by open.
- **Formula:** `(close-open)/(open+1e-12)`
- **Raw inputs:** `open`, `high`, `low`, `close`
- **Window / history:** 1
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `klen`

- **Purpose:** Daily high-low range scaled by open.
- **Formula:** `(high-low)/(open+1e-12)`
- **Raw inputs:** `open`, `high`, `low`, `close`
- **Window / history:** 1
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `kmid2`

- **Purpose:** Signed candle body scaled by total range.
- **Formula:** `(close-open)/(high-low+1e-12)`
- **Raw inputs:** `open`, `high`, `low`, `close`
- **Window / history:** 1
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `kup`

- **Purpose:** Upper shadow scaled by open.
- **Formula:** `(high-max(open,close))/(open+1e-12)`
- **Raw inputs:** `open`, `high`, `low`, `close`
- **Window / history:** 1
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `kup2`

- **Purpose:** Upper shadow as fraction of daily range.
- **Formula:** `(high-max(open,close))/(high-low+1e-12)`
- **Raw inputs:** `open`, `high`, `low`, `close`
- **Window / history:** 1
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `klow`

- **Purpose:** Lower shadow scaled by open.
- **Formula:** `(min(open,close)-low)/(open+1e-12)`
- **Raw inputs:** `open`, `high`, `low`, `close`
- **Window / history:** 1
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `klow2`

- **Purpose:** Lower shadow as fraction of daily range.
- **Formula:** `(min(open,close)-low)/(high-low+1e-12)`
- **Raw inputs:** `open`, `high`, `low`, `close`
- **Window / history:** 1
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `ksft`

- **Purpose:** Close location shift scaled by open.
- **Formula:** `(2*close-high-low)/(open+1e-12)`
- **Raw inputs:** `open`, `high`, `low`, `close`
- **Window / history:** 1
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `ksft2`

- **Purpose:** Close location shift scaled by daily range.
- **Formula:** `(2*close-high-low)/(high-low+1e-12)`
- **Raw inputs:** `open`, `high`, `low`, `close`
- **Window / history:** 1
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `gap_open`

- **Purpose:** open relative to previous close.
- **Formula:** `open_t / lag(close,1) - 1`
- **Raw inputs:** `open`, `close`
- **Window / history:** 1
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `gap_high`

- **Purpose:** high relative to previous close.
- **Formula:** `high_t / lag(close,1) - 1`
- **Raw inputs:** `high`, `close`
- **Window / history:** 1
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `gap_low`

- **Purpose:** low relative to previous close.
- **Formula:** `low_t / lag(close,1) - 1`
- **Raw inputs:** `low`, `close`
- **Window / history:** 1
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

## C. Trend and price position

### `madev_3`

- **Purpose:** Close deviation from 3-day simple moving average.
- **Formula:** `close_t / rolling_mean(close,3) - 1`
- **Raw inputs:** `close`
- **Window / history:** 3
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `madev_5`

- **Purpose:** Close deviation from 5-day simple moving average.
- **Formula:** `close_t / rolling_mean(close,5) - 1`
- **Raw inputs:** `close`
- **Window / history:** 5
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `madev_10`

- **Purpose:** Close deviation from 10-day simple moving average.
- **Formula:** `close_t / rolling_mean(close,10) - 1`
- **Raw inputs:** `close`
- **Window / history:** 10
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `madev_20`

- **Purpose:** Close deviation from 20-day simple moving average.
- **Formula:** `close_t / rolling_mean(close,20) - 1`
- **Raw inputs:** `close`
- **Window / history:** 20
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `madev_30`

- **Purpose:** Close deviation from 30-day simple moving average.
- **Formula:** `close_t / rolling_mean(close,30) - 1`
- **Raw inputs:** `close`
- **Window / history:** 30
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `madev_60`

- **Purpose:** Close deviation from 60-day simple moving average.
- **Formula:** `close_t / rolling_mean(close,60) - 1`
- **Raw inputs:** `close`
- **Window / history:** 60
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `emadev_5`

- **Purpose:** Close deviation from 5-day EMA.
- **Formula:** `close_t / ema(close,span=5,adjust=False,min_periods=5) - 1`
- **Raw inputs:** `close`
- **Window / history:** 5
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.
- **Implementation note:** EMA must be computed causally within ts_code.

### `emadev_10`

- **Purpose:** Close deviation from 10-day EMA.
- **Formula:** `close_t / ema(close,span=10,adjust=False,min_periods=10) - 1`
- **Raw inputs:** `close`
- **Window / history:** 10
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.
- **Implementation note:** EMA must be computed causally within ts_code.

### `emadev_20`

- **Purpose:** Close deviation from 20-day EMA.
- **Formula:** `close_t / ema(close,span=20,adjust=False,min_periods=20) - 1`
- **Raw inputs:** `close`
- **Window / history:** 20
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.
- **Implementation note:** EMA must be computed causally within ts_code.

### `emadev_60`

- **Purpose:** Close deviation from 60-day EMA.
- **Formula:** `close_t / ema(close,span=60,adjust=False,min_periods=60) - 1`
- **Raw inputs:** `close`
- **Window / history:** 60
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.
- **Implementation note:** EMA must be computed causally within ts_code.

### `ma_5_over_20`

- **Purpose:** Short/long SMA ratio (5/20).
- **Formula:** `rolling_mean(close,5) / rolling_mean(close,20) - 1`
- **Raw inputs:** `close`
- **Window / history:** 20
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `ma_10_over_20`

- **Purpose:** Short/long SMA ratio (10/20).
- **Formula:** `rolling_mean(close,10) / rolling_mean(close,20) - 1`
- **Raw inputs:** `close`
- **Window / history:** 20
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `ma_20_over_60`

- **Purpose:** Short/long SMA ratio (20/60).
- **Formula:** `rolling_mean(close,20) / rolling_mean(close,60) - 1`
- **Raw inputs:** `close`
- **Window / history:** 60
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `slope_5`

- **Purpose:** Normalized OLS slope of close over the last 5 trading days.
- **Formula:** `ols_slope(close over chronological x=0..4) / (close_t + 1e-12)`
- **Raw inputs:** `close`
- **Window / history:** 5
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.
- **Implementation note:** Use exactly w observations; chronological x increases toward t.

### `slope_10`

- **Purpose:** Normalized OLS slope of close over the last 10 trading days.
- **Formula:** `ols_slope(close over chronological x=0..9) / (close_t + 1e-12)`
- **Raw inputs:** `close`
- **Window / history:** 10
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.
- **Implementation note:** Use exactly w observations; chronological x increases toward t.

### `slope_20`

- **Purpose:** Normalized OLS slope of close over the last 20 trading days.
- **Formula:** `ols_slope(close over chronological x=0..19) / (close_t + 1e-12)`
- **Raw inputs:** `close`
- **Window / history:** 20
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.
- **Implementation note:** Use exactly w observations; chronological x increases toward t.

### `slope_60`

- **Purpose:** Normalized OLS slope of close over the last 60 trading days.
- **Formula:** `ols_slope(close over chronological x=0..59) / (close_t + 1e-12)`
- **Raw inputs:** `close`
- **Window / history:** 60
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.
- **Implementation note:** Use exactly w observations; chronological x increases toward t.

### `rsq_5`

- **Purpose:** OLS R-squared of close trend over the last 5 trading days.
- **Formula:** `ols_r_squared(close over chronological x=0..4)`
- **Raw inputs:** `close`
- **Window / history:** 5
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `rsq_10`

- **Purpose:** OLS R-squared of close trend over the last 10 trading days.
- **Formula:** `ols_r_squared(close over chronological x=0..9)`
- **Raw inputs:** `close`
- **Window / history:** 10
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `rsq_20`

- **Purpose:** OLS R-squared of close trend over the last 20 trading days.
- **Formula:** `ols_r_squared(close over chronological x=0..19)`
- **Raw inputs:** `close`
- **Window / history:** 20
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `rsq_60`

- **Purpose:** OLS R-squared of close trend over the last 60 trading days.
- **Formula:** `ols_r_squared(close over chronological x=0..59)`
- **Raw inputs:** `close`
- **Window / history:** 60
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `distmax_5`

- **Purpose:** Current close relative to rolling 5-day high.
- **Formula:** `close_t / rolling_max(high,5) - 1`
- **Raw inputs:** `close`, `high`
- **Window / history:** 5
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `distmax_20`

- **Purpose:** Current close relative to rolling 20-day high.
- **Formula:** `close_t / rolling_max(high,20) - 1`
- **Raw inputs:** `close`, `high`
- **Window / history:** 20
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `distmax_60`

- **Purpose:** Current close relative to rolling 60-day high.
- **Formula:** `close_t / rolling_max(high,60) - 1`
- **Raw inputs:** `close`, `high`
- **Window / history:** 60
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `distmin_5`

- **Purpose:** Current close relative to rolling 5-day low.
- **Formula:** `close_t / rolling_min(low,5) - 1`
- **Raw inputs:** `close`, `low`
- **Window / history:** 5
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `distmin_20`

- **Purpose:** Current close relative to rolling 20-day low.
- **Formula:** `close_t / rolling_min(low,20) - 1`
- **Raw inputs:** `close`, `low`
- **Window / history:** 20
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `distmin_60`

- **Purpose:** Current close relative to rolling 60-day low.
- **Formula:** `close_t / rolling_min(low,60) - 1`
- **Raw inputs:** `close`, `low`
- **Window / history:** 60
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `rsv_5`

- **Purpose:** Close position within rolling 5-day high-low range.
- **Formula:** `(close_t - rolling_min(low,5)) / (rolling_max(high,5) - rolling_min(low,5) + 1e-12)`
- **Raw inputs:** `close`, `high`, `low`
- **Window / history:** 5
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `rsv_10`

- **Purpose:** Close position within rolling 10-day high-low range.
- **Formula:** `(close_t - rolling_min(low,10)) / (rolling_max(high,10) - rolling_min(low,10) + 1e-12)`
- **Raw inputs:** `close`, `high`, `low`
- **Window / history:** 10
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `rsv_20`

- **Purpose:** Close position within rolling 20-day high-low range.
- **Formula:** `(close_t - rolling_min(low,20)) / (rolling_max(high,20) - rolling_min(low,20) + 1e-12)`
- **Raw inputs:** `close`, `high`, `low`
- **Window / history:** 20
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `rsv_60`

- **Purpose:** Close position within rolling 60-day high-low range.
- **Formula:** `(close_t - rolling_min(low,60)) / (rolling_max(high,60) - rolling_min(low,60) + 1e-12)`
- **Raw inputs:** `close`, `high`, `low`
- **Window / history:** 60
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

## D. Volatility

### `stdret_5`

- **Purpose:** Population standard deviation of 1-day returns over 5 days.
- **Formula:** `rolling_std(ret_1,5,ddof=0)`
- **Depends on:** `ret_1`
- **Window / history:** 5
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `stdret_10`

- **Purpose:** Population standard deviation of 1-day returns over 10 days.
- **Formula:** `rolling_std(ret_1,10,ddof=0)`
- **Depends on:** `ret_1`
- **Window / history:** 10
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `stdret_20`

- **Purpose:** Population standard deviation of 1-day returns over 20 days.
- **Formula:** `rolling_std(ret_1,20,ddof=0)`
- **Depends on:** `ret_1`
- **Window / history:** 20
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `stdret_60`

- **Purpose:** Population standard deviation of 1-day returns over 60 days.
- **Formula:** `rolling_std(ret_1,60,ddof=0)`
- **Depends on:** `ret_1`
- **Window / history:** 60
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `downvol_5`

- **Purpose:** Downside RMS return over 5 days.
- **Formula:** `sqrt(rolling_mean(min(ret_1,0)^2,5))`
- **Depends on:** `ret_1`
- **Window / history:** 5
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `downvol_10`

- **Purpose:** Downside RMS return over 10 days.
- **Formula:** `sqrt(rolling_mean(min(ret_1,0)^2,10))`
- **Depends on:** `ret_1`
- **Window / history:** 10
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `downvol_20`

- **Purpose:** Downside RMS return over 20 days.
- **Formula:** `sqrt(rolling_mean(min(ret_1,0)^2,20))`
- **Depends on:** `ret_1`
- **Window / history:** 20
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `downvol_60`

- **Purpose:** Downside RMS return over 60 days.
- **Formula:** `sqrt(rolling_mean(min(ret_1,0)^2,60))`
- **Depends on:** `ret_1`
- **Window / history:** 60
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `parkinson_5`

- **Purpose:** Parkinson high-low volatility over 5 days.
- **Formula:** `sqrt(rolling_mean(log(high/low)^2,5) / (4*ln(2)))`
- **Raw inputs:** `high`, `low`
- **Window / history:** 5
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.
- **Implementation note:** If high<=0 or low<=0, set daily term NaN.

### `parkinson_10`

- **Purpose:** Parkinson high-low volatility over 10 days.
- **Formula:** `sqrt(rolling_mean(log(high/low)^2,10) / (4*ln(2)))`
- **Raw inputs:** `high`, `low`
- **Window / history:** 10
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.
- **Implementation note:** If high<=0 or low<=0, set daily term NaN.

### `parkinson_20`

- **Purpose:** Parkinson high-low volatility over 20 days.
- **Formula:** `sqrt(rolling_mean(log(high/low)^2,20) / (4*ln(2)))`
- **Raw inputs:** `high`, `low`
- **Window / history:** 20
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.
- **Implementation note:** If high<=0 or low<=0, set daily term NaN.

### `parkinson_60`

- **Purpose:** Parkinson high-low volatility over 60 days.
- **Formula:** `sqrt(rolling_mean(log(high/low)^2,60) / (4*ln(2)))`
- **Raw inputs:** `high`, `low`
- **Window / history:** 60
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.
- **Implementation note:** If high<=0 or low<=0, set daily term NaN.

### `gkvol_5`

- **Purpose:** Garman-Klass OHLC volatility over 5 days.
- **Formula:** `sqrt(max(rolling_mean(0.5*log(high/low)^2 - (2*ln(2)-1)*log(close/open)^2,5),0))`
- **Raw inputs:** `open`, `high`, `low`, `close`
- **Window / history:** 5
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.
- **Implementation note:** Clamp rolling variance estimate at zero before sqrt; invalid non-positive prices -> NaN.

### `gkvol_10`

- **Purpose:** Garman-Klass OHLC volatility over 10 days.
- **Formula:** `sqrt(max(rolling_mean(0.5*log(high/low)^2 - (2*ln(2)-1)*log(close/open)^2,10),0))`
- **Raw inputs:** `open`, `high`, `low`, `close`
- **Window / history:** 10
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.
- **Implementation note:** Clamp rolling variance estimate at zero before sqrt; invalid non-positive prices -> NaN.

### `gkvol_20`

- **Purpose:** Garman-Klass OHLC volatility over 20 days.
- **Formula:** `sqrt(max(rolling_mean(0.5*log(high/low)^2 - (2*ln(2)-1)*log(close/open)^2,20),0))`
- **Raw inputs:** `open`, `high`, `low`, `close`
- **Window / history:** 20
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.
- **Implementation note:** Clamp rolling variance estimate at zero before sqrt; invalid non-positive prices -> NaN.

### `gkvol_60`

- **Purpose:** Garman-Klass OHLC volatility over 60 days.
- **Formula:** `sqrt(max(rolling_mean(0.5*log(high/low)^2 - (2*ln(2)-1)*log(close/open)^2,60),0))`
- **Raw inputs:** `open`, `high`, `low`, `close`
- **Window / history:** 60
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.
- **Implementation note:** Clamp rolling variance estimate at zero before sqrt; invalid non-positive prices -> NaN.

### `atr_5`

- **Purpose:** Average true range over 5 days normalized by current close.
- **Formula:** `rolling_mean(max(high-low, abs(high-lag(close,1)), abs(low-lag(close,1))),5) / (close_t + 1e-12)`
- **Raw inputs:** `high`, `low`, `close`
- **Window / history:** 5
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `atr_10`

- **Purpose:** Average true range over 10 days normalized by current close.
- **Formula:** `rolling_mean(max(high-low, abs(high-lag(close,1)), abs(low-lag(close,1))),10) / (close_t + 1e-12)`
- **Raw inputs:** `high`, `low`, `close`
- **Window / history:** 10
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `atr_20`

- **Purpose:** Average true range over 20 days normalized by current close.
- **Formula:** `rolling_mean(max(high-low, abs(high-lag(close,1)), abs(low-lag(close,1))),20) / (close_t + 1e-12)`
- **Raw inputs:** `high`, `low`, `close`
- **Window / history:** 20
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `atr_60`

- **Purpose:** Average true range over 60 days normalized by current close.
- **Formula:** `rolling_mean(max(high-low, abs(high-lag(close,1)), abs(low-lag(close,1))),60) / (close_t + 1e-12)`
- **Raw inputs:** `high`, `low`, `close`
- **Window / history:** 60
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

## E1. Volume

### `volratio_3`

- **Purpose:** Current volume divided by 3-day mean volume.
- **Formula:** `vol_t / (rolling_mean(vol,3) + 1e-12)`
- **Raw inputs:** `vol`
- **Window / history:** 3
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `volratio_5`

- **Purpose:** Current volume divided by 5-day mean volume.
- **Formula:** `vol_t / (rolling_mean(vol,5) + 1e-12)`
- **Raw inputs:** `vol`
- **Window / history:** 5
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `volratio_10`

- **Purpose:** Current volume divided by 10-day mean volume.
- **Formula:** `vol_t / (rolling_mean(vol,10) + 1e-12)`
- **Raw inputs:** `vol`
- **Window / history:** 10
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `volratio_20`

- **Purpose:** Current volume divided by 20-day mean volume.
- **Formula:** `vol_t / (rolling_mean(vol,20) + 1e-12)`
- **Raw inputs:** `vol`
- **Window / history:** 20
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `volratio_60`

- **Purpose:** Current volume divided by 60-day mean volume.
- **Formula:** `vol_t / (rolling_mean(vol,60) + 1e-12)`
- **Raw inputs:** `vol`
- **Window / history:** 60
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `volchg_1`

- **Purpose:** Log-volume change over 1 trading days.
- **Formula:** `log1p(vol_t) - log1p(lag(vol,1))`
- **Raw inputs:** `vol`
- **Window / history:** 1
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `volchg_5`

- **Purpose:** Log-volume change over 5 trading days.
- **Formula:** `log1p(vol_t) - log1p(lag(vol,5))`
- **Raw inputs:** `vol`
- **Window / history:** 5
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `volchg_20`

- **Purpose:** Log-volume change over 20 trading days.
- **Formula:** `log1p(vol_t) - log1p(lag(vol,20))`
- **Raw inputs:** `vol`
- **Window / history:** 20
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `logvol_std_5`

- **Purpose:** Population standard deviation of log1p(volume) over 5 days.
- **Formula:** `rolling_std(log1p(vol),5,ddof=0)`
- **Raw inputs:** `vol`
- **Window / history:** 5
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `logvol_std_20`

- **Purpose:** Population standard deviation of log1p(volume) over 20 days.
- **Formula:** `rolling_std(log1p(vol),20,ddof=0)`
- **Raw inputs:** `vol`
- **Window / history:** 20
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `logvol_std_60`

- **Purpose:** Population standard deviation of log1p(volume) over 60 days.
- **Formula:** `rolling_std(log1p(vol),60,ddof=0)`
- **Raw inputs:** `vol`
- **Window / history:** 60
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

## E2. Trading amount

### `amtratio_3`

- **Purpose:** Current amount divided by 3-day mean amount.
- **Formula:** `amount_t / (rolling_mean(amount,3) + 1e-12)`
- **Raw inputs:** `amount`
- **Window / history:** 3
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `amtratio_5`

- **Purpose:** Current amount divided by 5-day mean amount.
- **Formula:** `amount_t / (rolling_mean(amount,5) + 1e-12)`
- **Raw inputs:** `amount`
- **Window / history:** 5
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `amtratio_10`

- **Purpose:** Current amount divided by 10-day mean amount.
- **Formula:** `amount_t / (rolling_mean(amount,10) + 1e-12)`
- **Raw inputs:** `amount`
- **Window / history:** 10
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `amtratio_20`

- **Purpose:** Current amount divided by 20-day mean amount.
- **Formula:** `amount_t / (rolling_mean(amount,20) + 1e-12)`
- **Raw inputs:** `amount`
- **Window / history:** 20
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `amtratio_60`

- **Purpose:** Current amount divided by 60-day mean amount.
- **Formula:** `amount_t / (rolling_mean(amount,60) + 1e-12)`
- **Raw inputs:** `amount`
- **Window / history:** 60
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `amtchg_1`

- **Purpose:** Log-amount change over 1 trading days.
- **Formula:** `log1p(amount_t) - log1p(lag(amount,1))`
- **Raw inputs:** `amount`
- **Window / history:** 1
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `amtchg_5`

- **Purpose:** Log-amount change over 5 trading days.
- **Formula:** `log1p(amount_t) - log1p(lag(amount,5))`
- **Raw inputs:** `amount`
- **Window / history:** 5
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `amtchg_20`

- **Purpose:** Log-amount change over 20 trading days.
- **Formula:** `log1p(amount_t) - log1p(lag(amount,20))`
- **Raw inputs:** `amount`
- **Window / history:** 20
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `logamt_std_5`

- **Purpose:** Population standard deviation of log1p(amount) over 5 days.
- **Formula:** `rolling_std(log1p(amount),5,ddof=0)`
- **Raw inputs:** `amount`
- **Window / history:** 5
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `logamt_std_20`

- **Purpose:** Population standard deviation of log1p(amount) over 20 days.
- **Formula:** `rolling_std(log1p(amount),20,ddof=0)`
- **Raw inputs:** `amount`
- **Window / history:** 20
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `logamt_std_60`

- **Purpose:** Population standard deviation of log1p(amount) over 60 days.
- **Formula:** `rolling_std(log1p(amount),60,ddof=0)`
- **Raw inputs:** `amount`
- **Window / history:** 60
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

## E3. Volume-price relation

### `corr_ret_dlogvol_5`

- **Purpose:** Rolling correlation between ret_1 and one-day log-volume change over 5 days.
- **Formula:** `rolling_corr(ret_1, volchg_1,5)`
- **Depends on:** `ret_1`, `volchg_1`
- **Window / history:** 5
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `corr_ret_dlogvol_20`

- **Purpose:** Rolling correlation between ret_1 and one-day log-volume change over 20 days.
- **Formula:** `rolling_corr(ret_1, volchg_1,20)`
- **Depends on:** `ret_1`, `volchg_1`
- **Window / history:** 20
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `corr_ret_dlogamt_5`

- **Purpose:** Rolling correlation between ret_1 and one-day log-amount change over 5 days.
- **Formula:** `rolling_corr(ret_1, amtchg_1,5)`
- **Depends on:** `ret_1`, `amtchg_1`
- **Window / history:** 5
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `corr_ret_dlogamt_20`

- **Purpose:** Rolling correlation between ret_1 and one-day log-amount change over 20 days.
- **Formula:** `rolling_corr(ret_1, amtchg_1,20)`
- **Depends on:** `ret_1`, `amtchg_1`
- **Window / history:** 20
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

## F. Liquidity

### `illiq_1`

- **Purpose:** Log-scaled Amihud-style one-day illiquidity.
- **Formula:** `log1p(1e8 * abs(ret_1) / amount_t) if amount_t>0 else NaN`
- **Raw inputs:** `amount`
- **Depends on:** `ret_1`
- **Window / history:** 1
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `illiq_5`

- **Purpose:** 5-day mean of illiq_1.
- **Formula:** `rolling_mean(illiq_1,5)`
- **Depends on:** `illiq_1`
- **Window / history:** 5
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `illiq_20`

- **Purpose:** 20-day mean of illiq_1.
- **Formula:** `rolling_mean(illiq_1,20)`
- **Depends on:** `illiq_1`
- **Window / history:** 20
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `illiq_60`

- **Purpose:** 60-day mean of illiq_1.
- **Formula:** `rolling_mean(illiq_1,60)`
- **Depends on:** `illiq_1`
- **Window / history:** 60
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

## G. Limit-up / limit-down

### `flag_limit_up`

- **Purpose:** Official same-day limit-up flag.
- **Formula:** `raw flag_limit_up`
- **Raw inputs:** `flag_limit_up`
- **Window / history:** 1
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `flag_limit_down`

- **Purpose:** Official same-day limit-down flag.
- **Formula:** `raw flag_limit_down`
- **Raw inputs:** `flag_limit_down`
- **Window / history:** 1
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `limit_up_count_5`

- **Purpose:** Number of limit-up days in trailing 5 trading days, including t.
- **Formula:** `rolling_sum(flag_limit_up,5)`
- **Raw inputs:** `flag_limit_up`
- **Window / history:** 5
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `limit_up_count_20`

- **Purpose:** Number of limit-up days in trailing 20 trading days, including t.
- **Formula:** `rolling_sum(flag_limit_up,20)`
- **Raw inputs:** `flag_limit_up`
- **Window / history:** 20
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `limit_up_count_60`

- **Purpose:** Number of limit-up days in trailing 60 trading days, including t.
- **Formula:** `rolling_sum(flag_limit_up,60)`
- **Raw inputs:** `flag_limit_up`
- **Window / history:** 60
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `limit_down_count_5`

- **Purpose:** Number of limit-down days in trailing 5 trading days, including t.
- **Formula:** `rolling_sum(flag_limit_down,5)`
- **Raw inputs:** `flag_limit_down`
- **Window / history:** 5
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `limit_down_count_20`

- **Purpose:** Number of limit-down days in trailing 20 trading days, including t.
- **Formula:** `rolling_sum(flag_limit_down,20)`
- **Raw inputs:** `flag_limit_down`
- **Window / history:** 20
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `limit_down_count_60`

- **Purpose:** Number of limit-down days in trailing 60 trading days, including t.
- **Formula:** `rolling_sum(flag_limit_down,60)`
- **Raw inputs:** `flag_limit_down`
- **Window / history:** 60
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `days_since_limit_up`

- **Purpose:** Trading days since most recent limit-up, including 0 if today; capped at 61.
- **Formula:** `min(trading_days_since_last(flag_limit_up==1),61)`
- **Raw inputs:** `flag_limit_up`
- **Window / history:** 61
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.
- **Implementation note:** If no event is observed in available history, use 61.

### `days_since_limit_down`

- **Purpose:** Trading days since most recent limit-down, including 0 if today; capped at 61.
- **Formula:** `min(trading_days_since_last(flag_limit_down==1),61)`
- **Raw inputs:** `flag_limit_down`
- **Window / history:** 61
- **Scope:** `ts_code`
- **Missing rule:** NaN if required history/input is unavailable or invalid.
- **Implementation note:** If no event is observed in available history, use 61.

## H. Market state

### `market_mean_ret_1`

- **Purpose:** Cross-sectional mean 1-day return on trade_date.
- **Formula:** `mean_i(ret_1[i,t])`
- **Depends on:** `ret_1`
- **Scope:** `trade_date`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `market_median_ret_1`

- **Purpose:** Cross-sectional median 1-day return.
- **Formula:** `median_i(ret_1[i,t])`
- **Depends on:** `ret_1`
- **Scope:** `trade_date`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `market_median_ret_5`

- **Purpose:** Cross-sectional median 5-day return.
- **Formula:** `median_i(ret_5[i,t])`
- **Depends on:** `ret_5`
- **Scope:** `trade_date`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `market_median_ret_20`

- **Purpose:** Cross-sectional median 20-day return.
- **Formula:** `median_i(ret_20[i,t])`
- **Depends on:** `ret_20`
- **Scope:** `trade_date`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `market_breadth_up`

- **Purpose:** Fraction of stocks with positive 1-day return.
- **Formula:** `mean_i(1[ret_1[i,t] > 0]) over finite ret_1 only`
- **Depends on:** `ret_1`
- **Scope:** `trade_date`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `market_dispersion_ret_1`

- **Purpose:** Cross-sectional population std of 1-day return.
- **Formula:** `std_i(ret_1[i,t],ddof=0)`
- **Depends on:** `ret_1`
- **Scope:** `trade_date`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `market_dispersion_ret_5`

- **Purpose:** Cross-sectional population std of 5-day return.
- **Formula:** `std_i(ret_5[i,t],ddof=0)`
- **Depends on:** `ret_5`
- **Scope:** `trade_date`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `market_limit_up_ratio`

- **Purpose:** Fraction of stocks flagged limit-up on trade_date.
- **Formula:** `mean_i(flag_limit_up[i,t])`
- **Depends on:** `flag_limit_up`
- **Scope:** `trade_date`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `market_limit_down_ratio`

- **Purpose:** Fraction of stocks flagged limit-down on trade_date.
- **Formula:** `mean_i(flag_limit_down[i,t])`
- **Depends on:** `flag_limit_down`
- **Scope:** `trade_date`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `market_amount_ratio_20`

- **Purpose:** Total market amount relative to its trailing 20-day mean.
- **Formula:** `market_amount_t / (rolling_mean(market_amount,20) + 1e-12); market_amount_t=sum_i(amount[i,t])`
- **Raw inputs:** `amount`
- **Window / history:** 20
- **Scope:** `trade_date`
- **Missing rule:** NaN if required history/input is unavailable or invalid.
- **Implementation note:** First aggregate amount by trade_date, then compute causal 20-day rolling mean on the date series and join back.

## I. Stock-relative-to-market

### `relret_1`

- **Purpose:** Stock 1-day return minus same-date cross-sectional median.
- **Formula:** `ret_1[i,t] - median_i(ret_1[i,t])`
- **Depends on:** `ret_1`
- **Scope:** `trade_date`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `relret_5`

- **Purpose:** Stock 5-day return minus same-date cross-sectional median.
- **Formula:** `ret_5[i,t] - median_i(ret_5[i,t])`
- **Depends on:** `ret_5`
- **Scope:** `trade_date`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `relret_20`

- **Purpose:** Stock 20-day return minus same-date cross-sectional median.
- **Formula:** `ret_20[i,t] - median_i(ret_20[i,t])`
- **Depends on:** `ret_20`
- **Scope:** `trade_date`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `relret_60`

- **Purpose:** Stock 60-day return minus same-date cross-sectional median.
- **Formula:** `ret_60[i,t] - median_i(ret_60[i,t])`
- **Depends on:** `ret_60`
- **Scope:** `trade_date`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `relstd_20`

- **Purpose:** stdret_20 minus same-date cross-sectional median stdret_20.
- **Formula:** `stdret_20[i,t] - median_i(stdret_20[i,t])`
- **Depends on:** `stdret_20`
- **Scope:** `trade_date`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `relvolratio_20`

- **Purpose:** volratio_20 minus same-date cross-sectional median volratio_20.
- **Formula:** `volratio_20[i,t] - median_i(volratio_20[i,t])`
- **Depends on:** `volratio_20`
- **Scope:** `trade_date`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `relamtratio_20`

- **Purpose:** amtratio_20 minus same-date cross-sectional median amtratio_20.
- **Formula:** `amtratio_20[i,t] - median_i(amtratio_20[i,t])`
- **Depends on:** `amtratio_20`
- **Scope:** `trade_date`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

### `relilliq_20`

- **Purpose:** illiq_20 minus same-date cross-sectional median illiq_20.
- **Formula:** `illiq_20[i,t] - median_i(illiq_20[i,t])`
- **Depends on:** `illiq_20`
- **Scope:** `trade_date`
- **Missing rule:** NaN if required history/input is unavailable or invalid.

## J. Same-day cross-sectional ranks

### `csr_ret_1`

- **Purpose:** Same-date percentile rank of ret_1; average ties; finite values only.
- **Formula:** `rank_pct(ret_1 within trade_date, method='average')`
- **Depends on:** `ret_1`
- **Scope:** `trade_date`
- **Missing rule:** NaN where ret_1 is NaN; ranks computed only over finite same-date values.

### `csr_ret_3`

- **Purpose:** Same-date percentile rank of ret_3; average ties; finite values only.
- **Formula:** `rank_pct(ret_3 within trade_date, method='average')`
- **Depends on:** `ret_3`
- **Scope:** `trade_date`
- **Missing rule:** NaN where ret_3 is NaN; ranks computed only over finite same-date values.

### `csr_ret_5`

- **Purpose:** Same-date percentile rank of ret_5; average ties; finite values only.
- **Formula:** `rank_pct(ret_5 within trade_date, method='average')`
- **Depends on:** `ret_5`
- **Scope:** `trade_date`
- **Missing rule:** NaN where ret_5 is NaN; ranks computed only over finite same-date values.

### `csr_ret_10`

- **Purpose:** Same-date percentile rank of ret_10; average ties; finite values only.
- **Formula:** `rank_pct(ret_10 within trade_date, method='average')`
- **Depends on:** `ret_10`
- **Scope:** `trade_date`
- **Missing rule:** NaN where ret_10 is NaN; ranks computed only over finite same-date values.

### `csr_ret_20`

- **Purpose:** Same-date percentile rank of ret_20; average ties; finite values only.
- **Formula:** `rank_pct(ret_20 within trade_date, method='average')`
- **Depends on:** `ret_20`
- **Scope:** `trade_date`
- **Missing rule:** NaN where ret_20 is NaN; ranks computed only over finite same-date values.

### `csr_ret_60`

- **Purpose:** Same-date percentile rank of ret_60; average ties; finite values only.
- **Formula:** `rank_pct(ret_60 within trade_date, method='average')`
- **Depends on:** `ret_60`
- **Scope:** `trade_date`
- **Missing rule:** NaN where ret_60 is NaN; ranks computed only over finite same-date values.

### `csr_madev_5`

- **Purpose:** Same-date percentile rank of madev_5; average ties; finite values only.
- **Formula:** `rank_pct(madev_5 within trade_date, method='average')`
- **Depends on:** `madev_5`
- **Scope:** `trade_date`
- **Missing rule:** NaN where madev_5 is NaN; ranks computed only over finite same-date values.

### `csr_madev_20`

- **Purpose:** Same-date percentile rank of madev_20; average ties; finite values only.
- **Formula:** `rank_pct(madev_20 within trade_date, method='average')`
- **Depends on:** `madev_20`
- **Scope:** `trade_date`
- **Missing rule:** NaN where madev_20 is NaN; ranks computed only over finite same-date values.

### `csr_stdret_5`

- **Purpose:** Same-date percentile rank of stdret_5; average ties; finite values only.
- **Formula:** `rank_pct(stdret_5 within trade_date, method='average')`
- **Depends on:** `stdret_5`
- **Scope:** `trade_date`
- **Missing rule:** NaN where stdret_5 is NaN; ranks computed only over finite same-date values.

### `csr_stdret_20`

- **Purpose:** Same-date percentile rank of stdret_20; average ties; finite values only.
- **Formula:** `rank_pct(stdret_20 within trade_date, method='average')`
- **Depends on:** `stdret_20`
- **Scope:** `trade_date`
- **Missing rule:** NaN where stdret_20 is NaN; ranks computed only over finite same-date values.

### `csr_volratio_5`

- **Purpose:** Same-date percentile rank of volratio_5; average ties; finite values only.
- **Formula:** `rank_pct(volratio_5 within trade_date, method='average')`
- **Depends on:** `volratio_5`
- **Scope:** `trade_date`
- **Missing rule:** NaN where volratio_5 is NaN; ranks computed only over finite same-date values.

### `csr_volratio_20`

- **Purpose:** Same-date percentile rank of volratio_20; average ties; finite values only.
- **Formula:** `rank_pct(volratio_20 within trade_date, method='average')`
- **Depends on:** `volratio_20`
- **Scope:** `trade_date`
- **Missing rule:** NaN where volratio_20 is NaN; ranks computed only over finite same-date values.

### `csr_amtratio_5`

- **Purpose:** Same-date percentile rank of amtratio_5; average ties; finite values only.
- **Formula:** `rank_pct(amtratio_5 within trade_date, method='average')`
- **Depends on:** `amtratio_5`
- **Scope:** `trade_date`
- **Missing rule:** NaN where amtratio_5 is NaN; ranks computed only over finite same-date values.

### `csr_amtratio_20`

- **Purpose:** Same-date percentile rank of amtratio_20; average ties; finite values only.
- **Formula:** `rank_pct(amtratio_20 within trade_date, method='average')`
- **Depends on:** `amtratio_20`
- **Scope:** `trade_date`
- **Missing rule:** NaN where amtratio_20 is NaN; ranks computed only over finite same-date values.

### `csr_illiq_20`

- **Purpose:** Same-date percentile rank of illiq_20; average ties; finite values only.
- **Formula:** `rank_pct(illiq_20 within trade_date, method='average')`
- **Depends on:** `illiq_20`
- **Scope:** `trade_date`
- **Missing rule:** NaN where illiq_20 is NaN; ranks computed only over finite same-date values.

### `csr_rsv_20`

- **Purpose:** Same-date percentile rank of rsv_20; average ties; finite values only.
- **Formula:** `rank_pct(rsv_20 within trade_date, method='average')`
- **Depends on:** `rsv_20`
- **Scope:** `trade_date`
- **Missing rule:** NaN where rsv_20 is NaN; ranks computed only over finite same-date values.


## 7. Required implementation checks

Before Full147 is accepted:

- generated `Basic40` column count = 40;
- generated `Full147` column count = 147;
- generated names exactly match YAML membership;
- no duplicate feature names;
- per-stock rolling does not cross `ts_code`;
- synthetic future perturbations do not change features at earlier dates;
- same-day cross-sectional features do not use later dates;
- initial-history NaNs are expected and preserved;
- no infinite values remain.

## 8. Research use

Feature importance alone is not sufficient evidence for keeping a feature family. Later ablation decisions should use:

- single-factor daily Rank IC / ICIR where meaningful;
- decile/group monotonicity;
- change in official validation score;
- consistency across F1/F2/F3.

Any material feature-definition change requires a version bump and a decision record.

## 9. Frozen implementation clarifications (V1.0.1)

- All per-stock lag/rolling use ascending observed rows within ts_code.
- Ordinary rolling requires the complete window and exactly window finite values; otherwise NaN.
- Rolling correlation requires exactly w pairs of finite values inside the window; otherwise NaN.
- EMA uses min_periods=window.
- Same-date cross-sectional statistics use only finite values of that variable.
- market_breadth_up denominator is the count of finite ret_1 on that date.
- OLS R-squared is NaN when the dependent-variable window has zero variance.
- Do not insert absent trading-date rows, forward fill or backfill. Audit the actual panel first; do not redefine lag semantics.

These user-authorized clarifications override ambiguous earlier descriptions, without changing Basic40/Full147 membership.
