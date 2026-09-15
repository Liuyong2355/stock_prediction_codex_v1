"""Frozen E006/E007 models and supervised-only label/group preparation."""
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from .targets import training_target

EXPERIMENTS={'E006':('xgboost_reg_v1','rank'),'E007':('lightgbm_ranker_v1','rank_decile')}


def make_model(config,experiment):
    if experiment not in EXPERIMENTS: raise ValueError('Only E006/E007 are supported')
    model,target=EXPERIMENTS[experiment]
    entry=next(e for e in config['experiments'] if e['id']==experiment)
    assert (entry['feature_set'],entry['model'],entry['target'])==('Full147',model,target)
    cls=xgb.XGBRegressor if experiment=='E006' else lgb.LGBMRanker
    return cls(**config['model_catalog'][model]['params'])


def training_plan(dates,labels,experiment):
    """Input is exclusively the complete purged finite supervised training slice."""
    dates=np.asarray(dates); labels=np.asarray(labels,dtype=np.float64)
    if experiment not in EXPERIMENTS: raise ValueError('Only E006/E007 are supported')
    if dates.ndim!=1 or labels.ndim!=1 or len(dates)!=len(labels) or not len(labels):
        raise ValueError('Aligned nonempty training vectors required')
    if not np.isfinite(labels).all(): raise ValueError('Only finite supervised labels are allowed')
    if experiment=='E006':
        return np.arange(len(labels)),training_target(dates,labels,'rank'),None,None
    # Compute percentile ranks directly, avoiding roundoff from adding .5 to y_rank.
    pct=pd.Series(labels).groupby(dates,sort=False).rank(method='average',pct=True).to_numpy()
    relevance=np.minimum(np.floor(pct*10),9).astype(np.int32)
    order=np.argsort(dates,kind='stable')
    ordered_dates=dates[order]
    group_dates,counts=np.unique(ordered_dates,return_counts=True)
    counts=counts.astype(np.int32)
    assert int(counts.sum())==len(labels) and (counts>0).all()
    np.testing.assert_array_equal(np.repeat(group_dates,counts),ordered_dates)
    return order,relevance[order],counts,group_dates


def predict_batches(model,matrix,indices,names,batch_size=100_000):
    """Same float64 inputs, row order and batch size as E004, for both official APIs."""
    result=np.empty(len(indices),dtype=np.float64)
    for start in range(0,len(indices),batch_size):
        idx=indices[start:start+batch_size]
        result[start:start+len(idx)]=model.predict(pd.DataFrame(matrix[idx],columns=names,copy=False))
    return result


def iterations(model):
    return model.get_booster().num_boosted_rounds() if isinstance(model,xgb.XGBRegressor) else model.booster_.current_iteration()
