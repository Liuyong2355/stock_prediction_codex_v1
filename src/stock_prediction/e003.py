"""Run only frozen E003; invoke the unmodified organizer evaluator on saved predictions."""
import argparse
import csv
import json
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .baselines import configs, materialize, predict_batches, save_lightgbm_text, save_json, peak_memory_gb
from .build_basic40 import sha256
from .compare_official import load_official, replay
from .diagnose_e002 import top_sets, persistence, distribution
from .folds import split_fold

METRIC_MAP=dict(ic_mean='rank_ic_mean',ic_std='rank_ic_std',icir='icir',ic_positive_ratio='ic_positive_ratio',
                annual_excess='annualized_top_excess_return',top1_annual_ret='top1_annualized_absolute_return',
                mean_turnover='mean_turnover',final_score='final_score')


def official_daily_and_diagnostic(p):
    """Diagnostic mirror checked against actual organizer output, not an authority replacement."""
    selected,turnover,ties=top_sets(p)
    labeled_top,diagnostic_turnover,_=top_sets(p,True)
    previous=None; rows=[]
    for date,day in p.groupby('trade_date',sort=True):
        labeled=day.loc[day.y_ret_1d.notna()]
        ic=float(spearmanr(labeled.pred,labeled.y_ret_1d).statistic) if len(labeled)>=30 else np.nan
        eligible=labeled.loc[labeled.flag_limit_up==0]
        top=labeled_top.loc[labeled_top.trade_date==date]
        ret=float(top.y_ret_1d.mean()) if len(top) else np.nan
        bottom=eligible.sort_values('pred',ascending=False).iloc[-max(len(eligible)//10,1):]
        spread=ret-float(bottom.y_ret_1d.mean()) if len(eligible)>=100 else np.nan
        current=set(selected.loc[selected.trade_date==date,'ts_code']) if (day.flag_limit_up==0).sum()>=100 else None
        daily_turn=1-len(current & previous)/len(current | previous) if current is not None and previous else np.nan
        previous=current
        rows.append(dict(trade_date=int(date),rank_ic=ic,top_return=ret,
                         top_excess=ret-float(eligible.y_ret_1d.mean()) if len(eligible)>=100 else np.nan,
                         turnover=daily_turn,auxiliary_floor_top_bottom_spread=spread))
    missing=selected.loc[selected.y_ret_1d.isna()]
    freq=missing.groupby(['trade_date','pred']).size()
    diag=dict(official_turnover=turnover,diagnostic_exclude_missing_y_turnover=diagnostic_turnover,
              diagnostic_minus_official=diagnostic_turnover-turnover,missing_y_rows=int(p.y_ret_1d.isna().sum()),
              top_rows=len(selected),missing_top_rows=len(missing),missing_share_of_top=len(missing)/len(selected) if len(selected) else np.nan,
              missing_top_prediction_distribution=distribution(missing.pred),
              missing_top_duplicate_fraction=float(freq.loc[freq>1].sum()/len(missing)) if len(missing) else np.nan,
              dates_boundary_splits_tie=int(ties.boundary_splits_tie.sum()),persistence=persistence(selected,sorted(p.trade_date.unique())))
    return pd.DataFrame(rows),diag,selected,ties


def make_e003(config):
    experiment=next(e for e in config['experiments'] if e['id']=='E003')
    assert experiment['feature_set']=='Full147' and experiment['target']=='raw' and experiment['model']=='lightgbm_reg_v1'
    return lgb.LGBMRegressor(**config['model_catalog']['lightgbm_reg_v1']['params'])


def verify_inputs(root):
    manifest=json.loads((root/'outputs/full147_manifest.json').read_text(encoding='utf-8'))
    accepted=json.loads((root/'outputs/baselines/input_verification.json').read_text(encoding='utf-8'))
    assert manifest['feature_set']=='Full147' and len(manifest['feature_names'])==147 and manifest['data_version']==accepted['data_version']
    for item in manifest['datasets']['train']['files'].values(): assert sha256(item['path'])==item['sha256']
    assert sha256(root/'outputs/baselines/cache/labels.npy')==accepted['labels_sha256']
    assert sha256(root/'data/raw/训练集.csv')==accepted['raw_sha256']
    source=root/'reference/evaluate_official.py'
    assert source.read_bytes()==(root/'evaluate.py').read_bytes()
    prior=json.loads((root/'outputs/official_evaluator_comparison.json').read_text(encoding='utf-8'))
    assert sha256(source)==prior['script_sha256']
    protected=list((root/'outputs/baselines/E002').glob('*/*'))
    protected += [root/'evaluate.py',source,root/'outputs/official_evaluator_comparison.json',root/'outputs/e002_missing_label_diagnostic.json',root/'docs/FEATURE_SPEC_V1.md']
    protected += list((root/'config').glob('*.yaml'))
    save_json(root/'outputs/e003_input_verification.json',dict(data_version=manifest['data_version'],full147_files_verified=True,
             labels_hash_verified=True,raw_hash_verified=True,official_script_sha256=sha256(source),test_labels_used=False,
             protected_sha256={str(p.relative_to(root)):sha256(p) for p in protected if p.is_file()}))


def run_fold(root,fold_id):
    foldconfig,config=configs(root)
    fold=next(f for f in foldconfig['folds'] if f['id']==fold_id)
    manifest=json.loads((root/'outputs/full147_manifest.json').read_text(encoding='utf-8'))
    names=manifest['feature_names']; source=manifest['datasets']['train']['files']
    matrix=np.load(source['matrix']['path'],mmap_mode='r',allow_pickle=False)
    keys=np.load(source['keys']['path'],mmap_mode='r',allow_pickle=False)
    labels=np.load(root/'outputs/baselines/cache/labels.npy',mmap_mode='r',allow_pickle=False)
    train_idx,valid_idx,split=split_fold(keys['trade_date'],labels,fold,foldconfig['purge_rule']['purge_trading_days'])
    prior=json.loads((root/'outputs/baselines/E002'/fold_id/'result.json').read_text(encoding='utf-8'))
    assert split==prior['split'] and config['model_catalog']['lightgbm_reg_v1']['params']==prior['configured_params']
    assert manifest['data_version']==prior['data_version']
    out=root/'outputs/baselines/E003'/fold_id; out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists(): raise FileExistsError('Completed E003 fold exists')
    print(f'START E003/{fold_id}: {len(train_idx):,} train; {len(valid_idx):,} valid; purge={split["purge_dates"]}',flush=True)
    started=time.perf_counter(); np.random.seed(config['seed'])
    training=materialize(matrix,train_idx,out/'training_work.npy')
    prep_seconds=time.perf_counter()-started
    model=make_e003(config)
    fit_start=time.perf_counter()
    model.fit(pd.DataFrame(training,columns=names,copy=False),np.asarray(labels[train_idx]))
    fit_seconds=time.perf_counter()-fit_start
    assert model.booster_.current_iteration()==config['model_catalog']['lightgbm_reg_v1']['params']['n_estimators']
    print(f'FIT E003/{fold_id}: {fit_seconds:.1f}s; predicting',flush=True)
    pred=predict_batches(model,matrix,valid_idx)
    assert np.isfinite(pred).all()
    joblib.dump({'model':model,'feature_names':names,'preprocessor':None},out/'model.joblib')
    save_lightgbm_text(model,out/'model.txt')
    training._mmap.close(); del training
    (out/'training_work.npy').unlink()
    p=pd.DataFrame(dict(ts_code=keys['ts_code'][valid_idx],trade_date=keys['trade_date'][valid_idx],pred=pred,
                        y_ret_1d=np.asarray(labels[valid_idx]),flag_limit_up=matrix[valid_idx,names.index('flag_limit_up')]))
    p.to_csv(out/'predictions.csv.gz',index=False,float_format='%.17g',compression={'method':'gzip','mtime':0})
    roundtrip=pd.read_csv(out/'predictions.csv.gz',float_precision='round_trip')
    np.testing.assert_array_equal(roundtrip.pred,pred)
    with tempfile.TemporaryDirectory(prefix='e003_official_') as temp:
        official=replay(roundtrip,temp,load_official(root))
    # Default CSV parser reproduces the original script's numeric/tie behavior.
    saved=pd.read_csv(out/'predictions.csv.gz')
    e002=pd.read_csv(root/'outputs/baselines/E002'/fold_id/'predictions.csv.gz')
    pd.testing.assert_frame_equal(saved[['ts_code','trade_date','y_ret_1d','flag_limit_up']],e002[['ts_code','trade_date','y_ret_1d','flag_limit_up']])
    daily,diag,selected,ties=official_daily_and_diagnostic(saved)
    assert np.isclose(daily.rank_ic.mean(),official['ic_mean'],atol=1e-14,rtol=0)
    assert np.isclose(daily.rank_ic.std(ddof=1),official['ic_std'],atol=1e-14,rtol=0)
    assert np.isclose(daily.top_excess.mean()*252,official['annual_excess'],atol=1e-12,rtol=0)
    assert np.isclose(daily.top_return.mean()*252,official['top1_annual_ret'],atol=1e-12,rtol=0)
    assert np.isclose(diag['official_turnover'],official['mean_turnover'],atol=1e-14,rtol=0)
    top_missing_indices=selected.loc[selected.y_ret_1d.isna()].index.to_numpy()
    for label,idx in [('missing_y',saved.index[saved.y_ret_1d.isna()].to_numpy()),('missing_y_top',top_missing_indices)]:
        counts=np.isfinite(matrix[valid_idx[idx]]).sum(axis=1)
        diag[label+'_full147_finite_features']=distribution(counts)
    sample_idx=np.unique(np.r_[np.linspace(0,len(valid_idx)-1,200,dtype=int),top_missing_indices[:20]])
    loaded=joblib.load(out/'model.joblib')['model']
    np.testing.assert_allclose(predict_batches(loaded,matrix,valid_idx[sample_idx]),pred[sample_idx],atol=1e-14,rtol=0)
    daily.to_csv(out/'daily_metrics.csv',index=False,float_format='%.17g')
    ties.to_csv(out/'top_diagnostic_daily.csv',index=False)
    save_json(out/'missing_label_diagnostic.json',diag)
    metrics={METRIC_MAP[k]:v for k,v in official.items()}
    # Not returned by official script: report separately and label the auxiliary convention.
    auxiliary=dict(rank_ic_std_ddof0=float(daily.rank_ic.std(ddof=0)),
                   top1_bottom1_annualized_spread=float(daily.auxiliary_floor_top_bottom_spread.mean()*252),
                   spread_status='auxiliary only; floor groups matching official Top sizing; not returned by organizer')
    result=dict(experiment_id='E003',fold_id=fold_id,run_id=f'E003_{fold_id}_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}',
                timestamp=datetime.now(timezone.utc).isoformat(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
                data_version=manifest['data_version'],feature_spec_version=manifest['feature_spec_version'],protocol_version=config['experiment_protocol_version'],
                feature_set='Full147',n_features=147,target='raw',model='lightgbm_reg_v1',seed=config['seed'],
                configured_params=config['model_catalog']['lightgbm_reg_v1']['params'],resolved_params=model.get_params(),actual_iterations=model.booster_.current_iteration(),
                split=split,train_rows=len(train_idx),valid_rows=len(valid_idx),supervised_fit_rows=len(train_idx),preprocessing=None,
                train_time_sec=fit_seconds,preprocessing_materialization_sec=prep_seconds,elapsed_seconds=time.perf_counter()-started,peak_memory_gb=peak_memory_gb(),
                evaluator_status='unmodified_organizer_script_validation_replay',official_script_sha256=sha256(root/'reference/evaluate_official.py'),
                metrics=metrics,official=official,auxiliary=auxiliary,missing_label_diagnostic=diag,
                validation=dict(e002_split_params_keys_labels_flags_equal=True,saved_prediction_roundtrip=True,official_daily_aggregation_matches=True,model_reload_predictions_match=True),
                environment=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,lightgbm=lgb.__version__),
                config_sha256={p.name:sha256(p) for p in (root/'config').glob('*.yaml')},
                feature_manifest_sha256=sha256(root/'outputs/full147_manifest.json'),
                artifact_files={p.name:dict(sha256=sha256(p),bytes=p.stat().st_size) for p in out.iterdir() if p.is_file()})
    save_json(out/'result.json',result)
    fields=config['experiment_log']['required_columns']; row={k:result.get(k,'') for k in fields}
    row.update({k:split[k] for k in ['train_start','train_end','valid_start','valid_end']}); row.update(metrics)
    row['top1_bottom1_annualized_spread']=auxiliary['top1_bottom1_annualized_spread']
    row['params_json']=json.dumps(result['resolved_params'],sort_keys=True); row['dropped_all_nan_features_json']='[]'
    row['notes']='Official original-script validation replay; IC std ddof=1; frozen ddof=0 separately in auxiliary; spread is auxiliary floor-group convention; missing-y turnover effect preserved'
    with (root/config['experiment_log']['path']).open('a',encoding='utf-8',newline='') as stream:
        csv.DictWriter(stream,fieldnames=fields).writerow(row)
    print(f'DONE E003/{fold_id}: official Score={official["final_score"]:.9f}; diagnostic turnover={diag["diagnostic_exclude_missing_y_turnover"]:.6f}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--fold',choices=['F1','F2','F3']); args=parser.parse_args()
    root=Path.cwd()
    if args.fold: run_fold(root,args.fold)
    else:
        verify_inputs(root)
        for fold in ['F1','F2','F3']:
            subprocess.run([sys.executable,'-m','stock_prediction.e003','--fold',fold],cwd=root,check=True)


if __name__=='__main__': main()
