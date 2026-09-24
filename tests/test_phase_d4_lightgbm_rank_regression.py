import numpy as np
import pandas as pd

from stock_prediction.phase_d4_lightgbm_rank_regression import d4_complementarity, make_lightgbm_model


def test_d4_uses_frozen_e004_lightgbm_parameters():
    params = {"objective": "regression", "n_estimators": 800, "learning_rate": 0.03,
              "num_leaves": 63, "random_state": 42}
    experiments = {"experiments": [{"id": "E004", "feature_set": "Full147",
                                     "model": "lightgbm_reg_v1", "target": "rank"}],
                   "model_catalog": {"lightgbm_reg_v1": {"params": params}}}
    model, actual = make_lightgbm_model(experiments)
    assert actual == params
    assert model.get_params()["num_leaves"] == 63
    assert model.get_params()["learning_rate"] == 0.03


def test_d4_complementarity_reports_exclusive_fractions():
    codes = list("abcdefghij")
    baseline = pd.DataFrame({"ts_code": codes, "trade_date": [1] * 10, "pred": np.arange(10.0)})
    candidate = baseline.assign(pred=np.arange(10.0)[::-1], flag_limit_up=0)
    metrics, daily = d4_complementarity(candidate, baseline)
    assert metrics["top10_jaccard"]["mean"] == 0.0
    assert metrics["top10_exclusive_union_fraction"]["mean"] == 1.0
    assert metrics["top10_per_model_exclusive_fraction"]["mean"] == 1.0
    assert daily["intersection"].iloc[0] == 0
