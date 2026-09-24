import numpy as np
from scipy.stats import rankdata

from stock_prediction.audit_d5_partial import daily_partial_spearman


def test_partial_spearman_matches_independent_rank_residualization():
    x = np.array([1, 2, 3, 4, 5, 7, np.nan], dtype=float)
    y = np.array([2, 3, 1, 5, 4, 6, 9], dtype=float)
    prediction = np.array([1, 4, 2, 5, 3, 6, 8], dtype=float)
    actual = daily_partial_spearman(x[:, None], y, prediction, minimum_rows=3)[0, 3]
    xr, yr, pr = (rankdata(values[:6]) for values in (x, y, prediction))
    design = np.column_stack((np.ones(6), pr))
    x_residual = xr - design @ np.linalg.lstsq(design, xr, rcond=None)[0]
    y_residual = yr - design @ np.linalg.lstsq(design, yr, rcond=None)[0]
    expected = np.corrcoef(x_residual, y_residual)[0, 1]
    np.testing.assert_allclose(actual, expected, atol=1e-12)
