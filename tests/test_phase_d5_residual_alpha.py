import numpy as np
from scipy.stats import spearmanr

from stock_prediction.phase_d5_residual_alpha import exact_column_spearman, residual_and_regions


def test_exact_column_spearman_matches_scipy_with_pairwise_missing():
    values = np.array([[1, 5], [2, np.nan], [2, 3], [4, 1], [np.nan, 0]], dtype=float)
    residual = np.array([4, 3, 2, 1, 0], dtype=float)
    actual = exact_column_spearman(values, residual, minimum_rows=3)
    expected = [spearmanr(values[:, i], residual, nan_policy="omit").statistic for i in range(2)]
    np.testing.assert_allclose(actual, expected)


def test_residual_uses_average_percentile_ranks_and_fixed_regions():
    y = np.array([1.0, 2.0, 2.0, 4.0, 5.0])
    pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    residual, prediction_rank, regions = residual_and_regions(y, pred)
    np.testing.assert_allclose(residual, [-0.8, -0.3, -0.1, 0.4, 0.8])
    np.testing.assert_allclose(prediction_rank, [1.0, 0.8, 0.6, 0.4, 0.2])
    assert regions["top20"].tolist() == [True, False, False, False, False]
    assert regions["middle60"].tolist() == [False, True, True, True, False]
