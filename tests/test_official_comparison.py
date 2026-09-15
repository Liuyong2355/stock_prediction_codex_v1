from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from stock_prediction.compare_official import load_official, replay
from stock_prediction.evaluator import evaluate


def fixture(n=101):
    return pd.DataFrame([dict(ts_code=f"s{i:03}", trade_date=date, pred=float(i),
                              y_ret_1d=i / 1000, flag_limit_up=0)
                         for date in [20220104, 20220105] for i in range(n)])


def test_official_floor_and_preserved_inputs(tmp_path):
    root = Path(__file__).resolve().parents[1]
    assert (root / "evaluate.py").read_bytes() == (root / "reference/evaluate_official.py").read_bytes()
    frame = fixture()
    before = frame.copy(deep=True)
    official = replay(frame, tmp_path, load_official(root))
    assert official["annual_excess"] == pytest.approx((.0955 - .05) * 252)
    provisional, _ = evaluate(frame)
    assert provisional["metrics"]["annualized_top_excess_return"] != pytest.approx(official["annual_excess"])
    pd.testing.assert_frame_equal(frame, before)


def test_official_nan_prediction_propagates(tmp_path):
    frame = fixture()
    frame.loc[0, "pred"] = np.nan
    official = replay(frame, tmp_path, load_official(Path(__file__).resolve().parents[1]))
    assert np.isnan(official["ic_mean"]) and np.isnan(official["final_score"])
    assert np.isfinite(evaluate(frame)[0]["metrics"]["final_score"])


def test_official_minimum_thresholds(tmp_path):
    frame = fixture(20)
    with pytest.warns(RuntimeWarning):
        official = replay(frame, tmp_path, load_official(Path(__file__).resolve().parents[1]))
    assert np.isnan(official["ic_mean"])
    assert np.isnan(official["annual_excess"])
    assert np.isnan(official["mean_turnover"])
