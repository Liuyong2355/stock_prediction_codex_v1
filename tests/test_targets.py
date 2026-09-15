from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from stock_prediction.baselines import configs, predict_batches
from stock_prediction.e003 import make_full147_model
from stock_prediction.folds import split_fold
from stock_prediction.targets import training_target, array_sha256
from stock_prediction.compare_targets import independent_target, score_difference, summarize

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('target,expected',[
    ('raw',[8.,1.,3.,1.,-4.,5.]),
    ('rank',[.5,0.,.5,0.,0.,.5]),
    ('relative',[6.,0.,2.,0.,-6.,0.]),
])
def test_exact_targets_ties_singleton_interleaved_dates(target,expected):
    # Date 10: [8,-4] -> median 2; date 20: [1,3,1] -> median 1.
    dates=np.array([10,20,20,20,10,30])
    labels=np.array([8.,1.,3.,1.,-4.,5.])
    before=labels.copy()
    np.testing.assert_array_equal(training_target(dates,labels,target),expected)
    np.testing.assert_array_equal(labels,before)


@pytest.mark.parametrize('target',['raw','rank','relative'])
def test_constant_day_and_unavailable_labels(target):
    result=training_target([1,1,1],[2.,2.,2.],target)
    np.testing.assert_allclose(result,{'raw':2.,'rank':1/6,'relative':0.}[target])
    for unavailable in [np.nan,np.inf,-np.inf]:
        with pytest.raises(ValueError,match='finite supervised'):
            training_target([1,1],[1.,unavailable],target)


@pytest.mark.parametrize('target',['rank','relative'])
def test_fold_purge_missing_filter_future_mutation_and_day_isolation(target):
    dates=np.repeat([20211228,20211229,20211230,20220104],3)
    labels=np.array([1.,np.nan,3.,2.,4.,4.,100.,200.,300.,5.,6.,7.])
    fold=dict(id='F1',train_start='2021-12-28',train_end='2021-12-30',valid_start='2022-01-01',valid_end='2022-12-31')
    train,valid,meta=split_fold(dates,labels,fold)
    np.testing.assert_array_equal(train,[0,2,3,4,5])
    assert meta['purge_dates']==[20211230]
    expected=training_target(dates[train],labels[train],target)
    changed=labels.copy(); changed[dates>=20211230]=np.nan
    train2,valid2,_=split_fold(dates,changed,fold)
    np.testing.assert_array_equal(train,train2)
    np.testing.assert_array_equal(valid,valid2)
    np.testing.assert_array_equal(expected,training_target(dates[train2],changed[train2],target))
    changed=labels.copy(); changed[dates==20211229]=[-100,1000,500]
    actual=training_target(dates[train],changed[train],target)
    np.testing.assert_array_equal(expected[:2],actual[:2])


def test_rank_and_relative_translation_invariance_and_order():
    dates=np.array([2,1,2,1]); labels=np.array([1.,4.,3.,2.])
    perm=np.array([3,0,2,1])
    for target in ['rank','relative']:
        y=training_target(dates,labels,target)
        np.testing.assert_array_equal(y,training_target(dates,labels+dates*100,target))
        np.testing.assert_array_equal(y[perm],training_target(dates[perm],labels[perm],target))
    assert array_sha256(labels)!=array_sha256(labels[::-1])


@pytest.mark.parametrize('experiment,target',[('E004','rank'),('E005','relative')])
def test_frozen_target_fit_smoke_repeat_and_reload(experiment,target,tmp_path):
    _,config=configs(ROOT)
    model=make_full147_model(config,experiment)
    assert model.get_params()==make_full147_model(config,'E003').get_params()
    rng=np.random.default_rng(42)
    x=rng.normal(size=(4000,147)); x[:200,3]=np.nan
    dates=np.repeat(np.arange(20),200)
    raw=.02*x[:,0]+rng.normal(scale=.01,size=len(x))
    y=training_target(dates,raw,target)
    names=[f'f{i}' for i in range(147)]
    frame=pd.DataFrame(x,columns=names)
    model.fit(frame,y)
    assert model.booster_.current_iteration()==800 and model.n_features_in_==147
    pred=predict_batches(model,x,np.arange(200))
    assert np.isfinite(pred).all()
    again=make_full147_model(config,experiment).fit(frame,y)
    np.testing.assert_array_equal(pred,predict_batches(again,x,np.arange(200)))
    joblib.dump(model,tmp_path/'model.joblib')
    np.testing.assert_array_equal(pred,predict_batches(joblib.load(tmp_path/'model.joblib'),x,np.arange(200)))
    assert model.feature_name_==names  # transformed y is never appended to X
    bad=deepcopy(config)
    next(e for e in bad['experiments'] if e['id']==experiment)['target']='raw'
    with pytest.raises(AssertionError): make_full147_model(bad,experiment)


def test_out_of_scope_rejected():
    _,config=configs(ROOT)
    for experiment in ['E006','E007']:
        with pytest.raises(ValueError): make_full147_model(config,experiment)
    with pytest.raises(ValueError): training_target([1],[1.],'rank_decile')


def test_checkout_text_restoration_requires_exact_accepted_content():
    import hashlib
    from stock_prediction.restore_text_contract import accepted_text_bytes
    for accepted,current in [(b'a\r\nb\r\n',b'a\nb\n'),(b'a\nb\n',b'a\r\nb\r\n')]:
        digest=hashlib.sha256(accepted).hexdigest()
        assert accepted_text_bytes(current,digest)==accepted
        assert accepted_text_bytes(accepted,digest)==accepted
        with pytest.raises(ValueError,match='Content differs'):
            accepted_text_bytes(current.replace(b'b',b'changed'),digest)


@pytest.mark.parametrize('target',['raw','rank','relative'])
def test_independent_oracle_on_tied_random_panel(target):
    rng=np.random.default_rng(84)
    dates=rng.integers(1,20,1000)
    labels=rng.integers(-20,20,1000).astype(float)/100
    np.testing.assert_array_equal(training_target(dates,labels,target),independent_target(dates,labels,target))


def test_score_decomposition_signs_and_worst_fold():
    a=dict(ic_mean=.1,annual_excess=.5,mean_turnover=.8,final_score=.25)
    b=dict(ic_mean=.2,annual_excess=.6,mean_turnover=.9,final_score=.29)
    delta=score_difference(a,b)
    assert delta['ic_contribution']==pytest.approx(.04)
    assert delta['excess_contribution']==pytest.approx(.03)
    assert delta['turnover_contribution']==pytest.approx(-.03)
    rows=[dict(experiment=e,fold=f,ic_mean=v,annual_excess=v,mean_turnover=v,
               final_score=v,diagnostic_turnover=v,missing_top_fraction=v)
          for e in ['E003','E004','E005'] for f,v in [('F1',.1),('F2',.2),('F3',.3)]]
    for summary in summarize(rows):
        assert summary['final_score']['mean']==pytest.approx(.2)
        assert summary['final_score']['worst_fold']=='F1'
        assert summary['mean_turnover']['worst_fold']=='F3'
        assert summary['final_score']['std_ddof0']==pytest.approx(np.std([.1,.2,.3],ddof=0))


@pytest.mark.parametrize('experiment,target',[('E004','rank'),('E005','relative')])
def test_runner_passes_only_purged_finite_training_labels(experiment,target,tmp_path,monkeypatch):
    import json
    import platform
    import lightgbm as lgb
    from stock_prediction import e003
    from stock_prediction.build_basic40 import sha256

    foldconfig,config=configs(ROOT)
    dates=np.repeat([20180102,20211230,20211231,20220104],3)
    labels=np.array([1.,np.nan,3.,2.,4.,4.,100.,200.,300.,5.,6.,7.])
    keys=np.zeros(len(dates),dtype=[('trade_date','i8'),('ts_code','U4')]); keys['trade_date']=dates
    keys['ts_code']=np.tile(['s1','s2','s3'],4)
    cache=tmp_path/'outputs/baselines/cache'; cache.mkdir(parents=True)
    np.save(cache/'labels.npy',labels); np.save(tmp_path/'keys.npy',keys)
    np.save(tmp_path/'matrix.npy',np.zeros((len(dates),147)))
    (tmp_path/'config').mkdir()
    for p in (ROOT/'config').glob('*.yaml'):
        (tmp_path/'config'/p.name).write_bytes(p.read_bytes())
    manifest=dict(feature_names=[f'f{i}' for i in range(147)],data_version='fixture',
                  datasets={'train':{'files':{'matrix':{'path':str(tmp_path/'matrix.npy')},'keys':{'path':str(tmp_path/'keys.npy')}}}})
    manifest_path=tmp_path/'outputs/full147_manifest.json'
    manifest_path.write_text(json.dumps(manifest),encoding='utf-8')
    train,_,split=split_fold(dates,labels,foldconfig['folds'][0])
    prior=dict(split=split,configured_params=config['model_catalog']['lightgbm_reg_v1']['params'],
               data_version='fixture',feature_manifest_sha256=sha256(manifest_path),
               config_sha256={p.name:sha256(p) for p in (tmp_path/'config').glob('*.yaml')},
               resolved_params=make_full147_model(config,'E003').get_params(),
               environment=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,lightgbm=lgb.__version__))
    folder=tmp_path/'outputs/baselines/E003/F1'; folder.mkdir(parents=True)
    (folder/'result.json').write_text(json.dumps(prior),encoding='utf-8')
    class ObservedTarget(Exception): pass
    def observe(actual_dates,actual_labels,actual_target):
        assert actual_target==target
        np.testing.assert_array_equal(actual_dates,dates[train])
        np.testing.assert_array_equal(actual_labels,labels[train])
        assert max(actual_dates)==20211230 and len(actual_labels)==5
        raise ObservedTarget
    monkeypatch.setattr(e003,'training_target',observe)
    with pytest.raises(ObservedTarget): e003.run_fold(tmp_path,'F1',experiment)
