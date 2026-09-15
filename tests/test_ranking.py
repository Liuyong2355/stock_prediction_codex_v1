from pathlib import Path
import json
import platform

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from scipy.stats import rankdata

from stock_prediction.baselines import configs
from stock_prediction.build_basic40 import sha256
from stock_prediction.compare_official import replay,load_official
from stock_prediction.e003 import official_daily_and_diagnostic
from stock_prediction.folds import split_fold
from stock_prediction.ranking import make_model,training_plan,predict_batches,iterations
from stock_prediction.targets import training_target

ROOT=Path(__file__).resolve().parents[1]


def test_decile_ties_singletons_boundaries_and_stable_groups():
    dates=np.array([2,1,2,1,2,3]); labels=np.array([1.,8.,1.,-4.,2.,5.])
    order,y,group,group_dates=training_plan(dates,labels,'E007')
    np.testing.assert_array_equal(order,[1,3,0,2,4,5])
    np.testing.assert_array_equal(y,[9,5,5,5,9,9])
    np.testing.assert_array_equal(group,[2,3,1])
    np.testing.assert_array_equal(np.repeat(group_dates,group),dates[order])
    _,y,_,_=training_plan(np.ones(20),np.arange(20.),'E007')
    np.testing.assert_array_equal(y,np.minimum(np.arange(1,21)//2,9))
    assert y.dtype==np.int32 and set(y)==set(range(10))
    _,y,_,_=training_plan([1]*4,[.1]*4,'E007')
    np.testing.assert_array_equal(y,[6]*4)


@pytest.mark.parametrize('experiment',['E006','E007'])
def test_independent_daily_oracle_and_row_alignment(experiment):
    rng=np.random.default_rng(27)
    dates=rng.integers(1,25,7000); labels=rng.integers(-100,100,7000)/1000
    before=labels.copy()
    order,y,group,_=training_plan(dates,labels,experiment)
    expected=np.empty(len(labels),dtype=np.float64)
    for date in np.unique(dates):
        idx=np.flatnonzero(dates==date); pct=rankdata(labels[idx],method='average')/len(idx)
        expected[idx]=pct-.5 if experiment=='E006' else np.minimum(np.floor(pct*10),9)
    np.testing.assert_array_equal(y,expected[order])
    np.testing.assert_array_equal(labels,before)
    if experiment=='E006':
        np.testing.assert_array_equal(order,np.arange(len(labels)))
        np.testing.assert_array_equal(y,training_target(dates,labels,'rank'))
        assert group is None
    else:
        offsets=np.r_[0,np.cumsum(group)]
        assert group.sum()==len(labels)
        for start,end in zip(offsets[:-1],offsets[1:]):
            assert len(np.unique(dates[order[start:end]]))==1
            assert (np.diff(order[start:end])>0).all()


@pytest.mark.parametrize('experiment',['E006','E007'])
def test_purge_future_label_mutation_and_nonfinite_rejection(experiment):
    dates=np.repeat([20211228,20211230,20211231,20220104],3)
    labels=np.array([1.,np.nan,3.,2.,4.,4.,100.,200.,300.,5.,6.,7.])
    fold=dict(id='F1',train_start='2021-12-28',train_end='2021-12-31',valid_start='2022-01-01',valid_end='2022-12-31')
    train,valid,meta=split_fold(dates,labels,fold)
    np.testing.assert_array_equal(train,[0,2,3,4,5]); assert meta['purge_dates']==[20211231]
    plan=training_plan(dates[train],labels[train],experiment)
    changed=labels.copy(); changed[dates>=20211231]=np.nan
    train2,valid2,_=split_fold(dates,changed,fold)
    np.testing.assert_array_equal(valid2,valid)
    plan2=training_plan(dates[train2],changed[train2],experiment)
    for a,b in zip(plan,plan2): np.testing.assert_array_equal(a,b)
    for bad in [np.nan,np.inf,-np.inf]:
        with pytest.raises(ValueError,match='finite supervised'): training_plan([1,1],[1.,bad],experiment)


@pytest.mark.parametrize('experiment,cls,model_id',[
    ('E006',xgb.XGBRegressor,'xgboost_reg_v1'),('E007',lgb.LGBMRanker,'lightgbm_ranker_v1')])
def test_official_frozen_model_repeat_reload_and_evaluator(experiment,cls,model_id,tmp_path):
    _,config=configs(ROOT)
    model=make_model(config,experiment)
    assert isinstance(model,cls)
    for name,value in config['model_catalog'][model_id]['params'].items(): assert model.get_params()[name]==value
    rng=np.random.default_rng(42)
    x=rng.normal(size=(4000,147)); x[:200,4]=np.nan
    dates=np.tile(np.arange(20),200)
    raw=.02*x[:,0]+rng.normal(scale=.01,size=len(x))
    order,y,group,_=training_plan(dates,raw,experiment)
    names=[f'f{i}' for i in range(147)]
    kwargs={} if group is None else {'group':group}
    frame=pd.DataFrame(x[order],columns=names)
    model.fit(frame,y,**kwargs)
    assert iterations(model)==800 and model.n_features_in_==147
    selected=np.arange(400)
    pred=predict_batches(model,x,selected,names)
    assert np.isfinite(pred).all()
    repeat=make_model(config,experiment).fit(frame,y,**kwargs)
    np.testing.assert_array_equal(pred,predict_batches(repeat,x,selected,names))
    joblib.dump(model,tmp_path/'model.joblib')
    np.testing.assert_array_equal(pred,predict_batches(joblib.load(tmp_path/'model.joblib'),x,selected,names))
    if experiment=='E006':
        native=xgb.Booster(); native.load_model(model.get_booster().save_raw(raw_format='ubj'))
        restored=native.predict(xgb.DMatrix(pd.DataFrame(x[selected],columns=names)))
    else:
        native=lgb.Booster(model_str=model.booster_.model_to_string()); restored=native.predict(x[selected])
    np.testing.assert_array_equal(pred,restored)
    # Evaluate real model predictions on raw returns, with missing labels retained.
    p=pd.DataFrame(dict(ts_code=[f's{i%200}' for i in selected],trade_date=selected//200+1,
                        pred=pred,y_ret_1d=raw[selected],flag_limit_up=0))
    p.loc[p.index%13==0,'y_ret_1d']=np.nan
    official=replay(p,tmp_path,load_official(ROOT))
    daily,diag,_,_=official_daily_and_diagnostic(pd.read_csv(tmp_path/'submission.csv').merge(
        pd.read_csv(tmp_path/'测试集_Y.csv'),on=['ts_code','trade_date']).merge(pd.read_csv(tmp_path/'测试集_X.csv'),on=['ts_code','trade_date']))
    assert daily.rank_ic.mean()==pytest.approx(official['ic_mean'])
    assert daily.top_excess.mean()*252==pytest.approx(official['annual_excess'])
    assert diag['official_turnover']==pytest.approx(official['mean_turnover'])
    assert official['final_score']==pytest.approx(.4*official['ic_mean']+.3*official['annual_excess']+.3*(1-official['mean_turnover']))


@pytest.mark.parametrize('experiment',['E006','E007'])
def test_runner_only_passes_supervised_training_slice(experiment,tmp_path,monkeypatch):
    from stock_prediction import e006_e007 as runner
    foldconfig,config=configs(ROOT)
    dates=np.repeat([20180102,20211230,20211231,20220104],3)
    labels=np.array([1.,np.nan,3.,2.,4.,4.,100.,200.,300.,5.,6.,7.])
    keys=np.zeros(len(dates),dtype=[('trade_date','i8'),('ts_code','U4')]); keys['trade_date']=dates
    cache=tmp_path/'outputs/baselines/cache'; cache.mkdir(parents=True)
    np.save(cache/'labels.npy',labels); np.save(tmp_path/'keys.npy',keys)
    np.save(tmp_path/'matrix.npy',np.zeros((len(dates),147)))
    (tmp_path/'config').mkdir()
    for p in (ROOT/'config').glob('*.yaml'): (tmp_path/'config'/p.name).write_bytes(p.read_bytes())
    manifest=dict(feature_names=[f'f{i}' for i in range(147)],data_version='fixture',datasets={'train':{'files':{
        'matrix':{'path':str(tmp_path/'matrix.npy')},'keys':{'path':str(tmp_path/'keys.npy')}}}})
    path=tmp_path/'outputs/full147_manifest.json'; path.write_text(json.dumps(manifest),encoding='utf-8')
    train,_,split=split_fold(dates,labels,foldconfig['folds'][0])
    prior=dict(split=split,data_version='fixture',feature_manifest_sha256=sha256(path),
               config_sha256={p.name:sha256(p) for p in (tmp_path/'config').glob('*.yaml')},
               environment=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,lightgbm=lgb.__version__))
    folder=tmp_path/'outputs/baselines/E004/F1'; folder.mkdir(parents=True)
    (folder/'result.json').write_text(json.dumps(prior),encoding='utf-8')
    class ObservedPlan(Exception): pass
    def observe(actual_dates,actual_labels,actual_experiment):
        assert actual_experiment==experiment
        np.testing.assert_array_equal(actual_dates,dates[train]); np.testing.assert_array_equal(actual_labels,labels[train])
        assert max(actual_dates)==20211230 and len(actual_labels)==5
        raise ObservedPlan
    monkeypatch.setattr(runner,'training_plan',observe)
    with pytest.raises(ObservedPlan): runner.run_fold(tmp_path,experiment,'F1')
