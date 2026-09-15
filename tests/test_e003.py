from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import lightgbm as lgb

from stock_prediction.baselines import configs, save_lightgbm_text
from stock_prediction.compare_official import load_official, replay
from stock_prediction.e003 import make_e003, official_daily_and_diagnostic

ROOT=Path(__file__).resolve().parents[1]


def test_e003_official_daily_parity_and_separate_label_diagnostic(tmp_path):
    rng=np.random.default_rng(42)
    p=pd.DataFrame([dict(ts_code=f's{i}',trade_date=d,pred=float(rng.normal()) if i>=15 else 10.,
                         y_ret_1d=float(rng.normal()/100) if i>=15 else np.nan,flag_limit_up=int(i==50))
                    for d in [1,2,3] for i in range(135)])
    official=replay(p,tmp_path,load_official(ROOT))
    daily,diagnostic,_,_=official_daily_and_diagnostic(p)
    assert daily.rank_ic.mean()==pytest.approx(official['ic_mean'])
    assert daily.rank_ic.std(ddof=1)==pytest.approx(official['ic_std'])
    assert daily.top_excess.mean()*252==pytest.approx(official['annual_excess'])
    assert daily.top_return.mean()*252==pytest.approx(official['top1_annual_ret'])
    assert diagnostic['official_turnover']==pytest.approx(official['mean_turnover'])
    assert diagnostic['diagnostic_exclude_missing_y_turnover']>diagnostic['official_turnover']


def test_frozen_e003_official_library_smoke_147_features(tmp_path):
    _,config=configs(ROOT)
    model=make_e003(config)
    assert isinstance(model,lgb.LGBMRegressor)
    for k,v in config['model_catalog']['lightgbm_reg_v1']['params'].items(): assert model.get_params()[k]==v
    rng=np.random.default_rng(42)
    x=pd.DataFrame(rng.normal(size=(4000,147)),columns=[f'f{i}' for i in range(147)])
    x.iloc[:200,3]=np.nan
    y=x.iloc[:,0]*.02+rng.normal(scale=.01,size=len(x))
    model.fit(x,y)
    assert model.booster_.current_iteration()==800 and model.n_features_in_==147
    pred=model.predict(x.iloc[:200])
    assert np.isfinite(pred).all()
    path=tmp_path/'模型.txt'; save_lightgbm_text(model,path)
    loaded=lgb.Booster(model_str=path.read_text(encoding='utf-8'))
    np.testing.assert_allclose(loaded.predict(x.iloc[:200]),pred,rtol=0,atol=1e-14)
