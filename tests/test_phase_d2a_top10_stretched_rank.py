import numpy as np
import pandas as pd
import pytest

from stock_prediction.phase_d2a_top10_stretched_rank import top10_stretched_rank_target


def test_top10_stretched_target_changes_only_daily_top_decile():
    dates = np.repeat([1, 2], 20)
    labels = np.r_[np.arange(20.0), np.arange(20.0)[::-1]]
    u = pd.Series(labels).groupby(dates, sort=False).rank(method="average", pct=True).to_numpy()
    actual = top10_stretched_rank_target(dates, labels)
    ordinary = u - 0.5
    np.testing.assert_array_equal(actual[u <= 0.9], ordinary[u <= 0.9])
    np.testing.assert_allclose(actual[u > 0.9], 0.9 + 10 * (u[u > 0.9] - 0.9) - 0.5)
    assert np.all(np.diff(actual[:20]) > 0)
    assert np.all(np.diff(actual[20:]) < 0)


def test_top10_stretched_target_uses_average_ties_within_date():
    actual = top10_stretched_rank_target([1] * 10, [0, 1, 2, 3, 4, 5, 6, 7, 9, 9])
    assert actual[-1] == actual[-2]
    assert actual[-1] == pytest.approx(0.9 + 10 * (0.95 - 0.9) - 0.5)


def test_top10_stretched_target_rejects_nonfinite_or_parameter_search():
    with pytest.raises(ValueError):
        top10_stretched_rank_target([1], [np.nan])
    with pytest.raises(ValueError):
        top10_stretched_rank_target([1], [1.0], coefficient=5.0)
