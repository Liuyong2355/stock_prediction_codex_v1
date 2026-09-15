import numpy as np
import pandas as pd
import pytest

from stock_prediction.diagnose_e002 import persistence, top_sets


def test_diagnostic_filter_is_separate_and_preserves_inputs():
    rows = []
    for date in [20220104, 20220105]:
        for i in range(120):
            rows.append(dict(ts_code=f"s{i}", trade_date=date, flag_limit_up=0,
                             y_ret_1d=np.nan if i < 12 else 0.,
                             pred=1000. if i < 12 else float(i if date == 20220104 else -i)))
    p = pd.DataFrame(rows)
    before = p.copy(deep=True)
    top, official, daily = top_sets(p)
    filtered, diagnostic, _ = top_sets(p, True)
    assert official == 0 and diagnostic == 1
    assert top.y_ret_1d.isna().all() and filtered.y_ret_1d.notna().all()
    assert len(filtered) == 20  # floor(108/10) per day
    assert (daily.cutoff_tie_group == 12).all()
    pd.testing.assert_frame_equal(p, before)


def test_missing_selection_run_breaks_on_market_date_gap():
    selected = pd.DataFrame(dict(ts_code=["a", "a", "a", "b"],
                                 trade_date=[1, 2, 4, 2], y_ret_1d=[np.nan] * 3 + [1.]))
    result = persistence(selected, [1, 2, 3, 4])
    assert result["stocks"] == 1 and result["runs"] == 2
    assert result["run_days"]["max"] == 2


def test_boundary_tie_detection_and_short_day_reset():
    p = pd.DataFrame([dict(ts_code=str(i), trade_date=d, pred=1., y_ret_1d=0., flag_limit_up=0)
                      for d, n in [(1, 101), (2, 99), (3, 101)] for i in range(n)])
    _, turnover, daily = top_sets(p)
    assert np.isnan(turnover)
    assert daily.boundary_splits_tie.all()
    assert (daily.top_rows == 10).all()
