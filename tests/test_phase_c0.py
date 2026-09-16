import numpy as np
import pandas as pd

from stock_prediction.phase_c0 import causal_ema, daily_rank, hysteresis_predictions, select_candidate, transform


def panel(days=(1, 2, 3), n=120):
    return pd.DataFrame([{"ts_code": f"s{i:03d}", "trade_date": d, "pred": float((i * (d + 1)) % 37),
                          "y_ret_1d": np.nan if i % 17 == 0 else i / 1000, "flag_limit_up": int(i == 119)}
                         for d in days for i in range(n)])


def test_daily_rank_is_same_date_average_and_label_independent():
    frame = panel(days=(1, 2), n=5)
    frame.loc[frame.trade_date == 2, "pred"] += 1000
    a = daily_rank(frame)
    changed = frame.copy(); changed.y_ret_1d = np.arange(len(frame)) * -999
    pd.testing.assert_series_equal(a, daily_rank(changed))
    assert a[frame.trade_date == 1].max() == 1
    assert a[frame.trade_date == 2].min() > 0


def test_ema_is_causal_future_safe_and_fold_local():
    frame = panel(n=8)
    ranks = daily_rank(frame)
    a = causal_ema(frame, ranks, .6)
    changed = ranks.copy(); changed[frame.trade_date == 3] = 999
    b = causal_ema(frame, changed, .6)
    np.testing.assert_array_equal(a[frame.trade_date < 3], b[frame.trade_date < 3])
    fold2 = frame.loc[frame.trade_date >= 2].reset_index(drop=True)
    local = causal_ema(fold2, daily_rank(fold2), .6)
    assert np.array_equal(local[fold2.trade_date == 2], daily_rank(fold2)[fold2.trade_date == 2])


def test_hysteresis_exact_top_limit_reset_ties_finite_deterministic():
    frame = panel(days=(1, 2, 3, 4), n=120)
    # Day 3 is below the official minimum and must break persistence.
    frame = frame.loc[~((frame.trade_date == 3) & (frame.ts_code >= "s090"))].reset_index(drop=True)
    smooth = daily_rank(frame)
    a, targets = hysteresis_predictions(frame, smooth, .15, return_top_sets=True)
    b = hysteresis_predictions(frame, smooth, .15)
    np.testing.assert_array_equal(a, b)
    assert np.isfinite(a).all()
    assert len(targets[1]) == len(frame[(frame.trade_date == 1) & (frame.flag_limit_up == 0)]) // 10
    assert "s119" not in targets[1]
    assert targets[3] == set()
    for _, values in pd.DataFrame({"date": frame.trade_date, "pred": a}).groupby("date"):
        assert values.pred.is_unique


def test_target_top_equals_official_sort_and_transform_repeatable():
    frame = panel()
    first, targets = transform(frame, .7, .125)
    second, _ = transform(frame, .7, .125)
    np.testing.assert_array_equal(first.pred, second.pred)
    for date, day in first.groupby("trade_date"):
        eligible = day.loc[day.flag_limit_up == 0].sort_values("pred", ascending=False)
        actual = set(eligible.iloc[:len(eligible) // 10].ts_code)
        assert actual == targets[int(date)]


def test_selection_tie_break_order():
    rows = []
    for alpha, exit_fraction, score, turnover in [(1., .15, .2, .3), (.9, .10, .2, .3)]:
        for fold in ["F1", "F2"]:
            rows.append({"alpha": alpha, "exit_fraction": exit_fraction, "final_score": score, "mean_turnover": turnover})
    assert select_candidate(rows)["alpha"] == 1.
