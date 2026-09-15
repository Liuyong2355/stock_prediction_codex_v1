import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import lightgbm as lgb

from stock_prediction.folds import split_fold
from stock_prediction.preprocessing import RidgePreprocessor
from stock_prediction.evaluator import evaluate, groups
from stock_prediction.baselines import configs, make_model, materialize, predict_batches, save_lightgbm_text
from pathlib import Path


def test_purge_observed_date_before_label_filter_and_validation_keeps_nan():
    fold = dict(id="F", train_start="2021-12-28", train_end="2021-12-31", valid_start="2022-01-01", valid_end="2022-12-31")
    dates = np.array([20211228,20211229,20211230,20211230,20220104,20220104,20220105])
    labels = np.array([1.,2.,np.nan,np.nan,3.,np.nan,4.])
    train, valid, info = split_fold(dates, labels, fold)
    assert train.tolist() == [0,1]
    assert valid.tolist() == [4,5,6]
    assert info["purge_dates"] == [20211230] and info["purge_rows"] == 2
    assert info["purge_finite_label_rows"] == 0
    assert info["valid_finite_label_rows"] == 2


def test_actual_config_folds_expand_and_do_not_train_in_validation():
    root = Path(__file__).resolve().parents[1]
    folds, _ = configs(root)
    dates = np.array([20180102,20201231,20211230,20211231,20220104,20221229,20221230,20230103,20231228,20231229,20240102,20241231])
    previous = set()
    for fold in folds["folds"]:
        train, valid, info = split_fold(dates, np.ones(len(dates)), fold)
        assert previous <= set(train)
        previous = set(train)
        assert dates[train].max() < dates[valid].min()
        assert not np.isin(dates[train], info["purge_dates"]).any()
        assert len(info["purge_dates"]) == 1


def test_overlap_rejected():
    fold = dict(id="F", train_start="2020-01-01", train_end="2022-01-04", valid_start="2022-01-01", valid_end="2022-12-31")
    with pytest.raises(ValueError, match="overlap"):
        split_fold([20200102,20210104,20220104], [1.,2.,3.], fold)


def test_ridge_preprocessing_only_sees_training_rows():
    matrix = np.array([[1.,np.nan,1.],[3.,np.nan,3.],[np.nan,np.nan,5.],[1e9,7.,1e9]])
    train = np.array([0,1,2])
    pre = RidgePreprocessor(["a","all_nan","b"], batch_size=2).fit(matrix, train)
    assert pre.dropped_features_ == ["all_nan"]
    assert pre.metadata()["median"] == [2.,3.]
    assert pre.metadata()["scaler_fit_rows"] == 3
    reference = SimpleImputer(strategy="median").fit_transform(matrix[train][:,[0,2]])
    expected = StandardScaler().fit_transform(reference)
    np.testing.assert_allclose(pre.transform(matrix[train]), expected, atol=1e-15)
    changed = matrix.copy(); changed[3] = [-1e12,-1e12,-1e12]
    again = RidgePreprocessor(["a","all_nan","b"], batch_size=2).fit(changed, train)
    assert pre.metadata() == again.metadata()
    original = matrix.copy()
    pre.transform(matrix[3:])
    np.testing.assert_array_equal(matrix, original)


def test_purged_row_cannot_supply_imputation_or_rescue_all_nan_column():
    fold = dict(id="F", train_start="2021-01-01", train_end="2021-12-31", valid_start="2022-01-01", valid_end="2022-12-31")
    dates = [20210104,20210105,20211231,20220104]
    matrix = np.array([[1.,np.nan],[3.,np.nan],[1000.,9.],[2000.,10.]])
    train, _, _ = split_fold(dates, [1.,2.,3.,4.], fold)
    pre = RidgePreprocessor(["a","b"]).fit(matrix, train)
    assert pre.metadata()["median"] == [2.]
    assert pre.dropped_features_ == ["b"]


def daily_panel(n=20, dates=(20240102,20240103)):
    return pd.DataFrame([dict(ts_code=f"{i:06d}.SZ",trade_date=d,pred=float(i),y_ret_1d=i/1000,flag_limit_up=0)
                         for d in dates for i in range(n)])


def test_evaluator_matches_hand_calculated_metrics():
    p = daily_panel()
    result, daily = evaluate(p)
    m = result["metrics"]
    assert m["rank_ic_mean"] == pytest.approx(1.)
    assert m["rank_ic_std"] == 0
    assert np.isnan(m["icir"])
    assert m["annualized_top_excess_return"] == pytest.approx(.009*252)
    assert m["mean_turnover"] == 0
    assert m["final_score"] == pytest.approx(.4+.3*.009*252+.3)
    assert m["top1_annualized_absolute_return"] == pytest.approx(.0185*252)
    assert m["top1_bottom1_annualized_spread"] == pytest.approx(.018*252)
    assert np.isnan(daily.turnover.iloc[0])
    assert result["evaluator_status"] == "provisional"


def test_evaluator_population_std_and_complete_top_replacement():
    p = daily_panel()
    p.loc[p.trade_date==20240103, "pred"] *= -1
    result, _ = evaluate(p)
    assert result["metrics"]["rank_ic_mean"] == pytest.approx(0.)
    assert result["metrics"]["rank_ic_std"] == pytest.approx(1.)
    assert result["metrics"]["ic_positive_ratio"] == .5
    assert result["metrics"]["mean_turnover"] == 1.


def test_turnover_ignores_missing_labels_but_excludes_limit_up():
    p = daily_panel()
    p.loc[p.ts_code=="000019.SZ", "y_ret_1d"] = np.nan
    p.loc[(p.ts_code=="000019.SZ") & (p.trade_date==20240103), "flag_limit_up"] = 1
    _, daily = evaluate(p)
    assert daily.ic_rows.tolist() == [19,19]
    assert daily.turnover_eligible_rows.tolist() == [20,19]
    assert daily.turnover.iloc[1] == pytest.approx(2/3)
    p["flag_limit_down"] = 1  # Dual flags remain governed by flag_limit_up only.
    _, same = evaluate(p)
    pd.testing.assert_frame_equal(daily, same)


def test_ties_group_sizes_and_nonfinite_predictions_are_explicit():
    p = daily_panel(21)
    p["pred"] = 0.
    top, bottom = groups(p.loc[p.trade_date==20240102].sample(frac=1, random_state=42))
    assert top.ts_code.tolist() == ["000000.SZ","000001.SZ","000002.SZ"]
    assert bottom.ts_code.tolist() == ["000019.SZ","000020.SZ"]
    p.loc[0,"pred"] = np.nan; p.loc[1,"pred"] = np.inf
    original = p.copy(deep=True)
    result, daily = evaluate(p)
    assert result["coverage"]["nonfinite_predictions"] == 2
    assert daily.ic_rows.iloc[0] == 19
    assert np.isnan(result["metrics"]["rank_ic_mean"])
    pd.testing.assert_frame_equal(p, original)


def test_small_dates_do_not_bridge_turnover_and_duplicates_fail():
    p = daily_panel(dates=(20240102,20240103,20240104))
    p = p.loc[~((p.trade_date==20240103) & (p.pred>=5))]
    _, daily = evaluate(p)
    assert daily.turnover.isna().all()
    with pytest.raises(ValueError, match="unique"):
        evaluate(pd.concat([p,p.iloc[:1]]))


@pytest.mark.parametrize("experiment", ["E001","E002"])
def test_official_model_smoke_with_frozen_configuration(tmp_path, experiment):
    _, config = configs(Path(__file__).resolve().parents[1])
    rng = np.random.default_rng(42)
    matrix = rng.normal(size=(4200,40)); matrix[0,0] = np.nan
    labels = np.nan_to_num(matrix[:,0]) * .01 + rng.normal(0,.001,len(matrix))
    train = np.arange(4000); valid = np.arange(4000,4200)
    pre = RidgePreprocessor([f"f{i}" for i in range(40)]).fit(matrix,train) if experiment=="E001" else None
    training = materialize(matrix,train,tmp_path/"train.npy",pre)
    model = make_model(experiment,config)
    assert isinstance(model, Ridge if experiment=="E001" else lgb.LGBMRegressor)
    catalog = "ridge_v1" if experiment=="E001" else "lightgbm_reg_v1"
    for key,value in config["model_catalog"][catalog]["params"].items():
        assert model.get_params()[key] == value
    model.fit(training,labels[train])
    prediction = predict_batches(model,matrix,valid,pre,batch_size=80)
    assert np.isfinite(prediction).all() and len(prediction)==200
    assert np.std(prediction)>0
    if experiment == "E002":
        destination = tmp_path / "中文模型.txt"
        save_lightgbm_text(model, destination)
        loaded = lgb.Booster(model_str=destination.read_text(encoding="utf-8"))
        np.testing.assert_allclose(loaded.predict(matrix[valid]), prediction)
    training._mmap.close()
