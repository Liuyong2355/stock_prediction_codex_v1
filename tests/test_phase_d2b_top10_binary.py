import numpy as np
import pandas as pd
import pytest

from stock_prediction.phase_d2b_top10_binary import complementarity, make_binary_model, top10_binary_target


def test_binary_target_is_strictly_above_daily_90th_percentile_rank():
    dates = np.repeat([1, 2], 20)
    labels = np.r_[np.arange(20.0), np.arange(20.0)[::-1]]
    actual = top10_binary_target(dates, labels)
    expected = np.tile([0] * 18 + [1] * 2, 2)
    expected[20:] = expected[20:][::-1]
    np.testing.assert_array_equal(actual, expected)


def test_binary_target_preserves_average_ties_at_boundary():
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 8]
    actual = top10_binary_target([1] * 10, labels)
    np.testing.assert_array_equal(actual, [0] * 8 + [1, 1])
    boundary_tie = top10_binary_target([1] * 10, [0, 1, 2, 3, 4, 5, 6, 7, 7, 9])
    np.testing.assert_array_equal(boundary_tie, [0] * 9 + [1])


def test_binary_target_rejects_nonfinite_and_threshold_search():
    with pytest.raises(ValueError):
        top10_binary_target([1], [np.nan])
    with pytest.raises(ValueError):
        top10_binary_target([1], [1.0], threshold=0.8)


def test_binary_model_changes_only_required_objective():
    experiments = {"experiments": [{"id": "E006", "feature_set": "Full147", "model": "xgboost_reg_v1", "target": "rank"}],
                   "model_catalog": {"xgboost_reg_v1": {"params": {"objective": "reg:squarederror", "n_estimators": 800, "random_state": 42}}}}
    _, params, changed = make_binary_model(experiments)
    assert params == {"objective": "binary:logistic", "n_estimators": 800, "random_state": 42}
    assert changed == {"objective": ("reg:squarederror", "binary:logistic")}


def test_complementarity_identical_and_disjoint_top_sets():
    base = pd.DataFrame({"ts_code": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
                         "trade_date": [1] * 10, "pred": np.arange(10.0)})
    candidate = base.assign(flag_limit_up=0)
    comp, _ = complementarity(candidate, base)
    assert comp["prediction_spearman"]["mean"] == pytest.approx(1.0)
    assert comp["top10_jaccard"]["mean"] == pytest.approx(1.0)
    reversed_candidate = candidate.assign(pred=np.arange(10.0)[::-1])
    comp, _ = complementarity(reversed_candidate, base)
    assert comp["prediction_spearman"]["mean"] == pytest.approx(-1.0)
    assert comp["top10_jaccard"]["mean"] == 0.0
