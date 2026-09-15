"""Execute only frozen E006/E007 using official model libraries and evaluator."""
import argparse
import csv
import json
import platform
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from datetime import datetime,timezone

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from .baselines import configs,materialize,save_json,save_lightgbm_text,peak_memory_gb
from .build_basic40 import sha256
from .compare_official import load_official,replay
from .diagnose_e002 import distribution
from .e003 import METRIC_MAP,official_daily_and_diagnostic
from .folds import split_fold
from .ranking import EXPERIMENTS,make_model,training_plan,predict_batches,iterations
from .targets import array_sha256

BASELINE='24e979d361516af6d3d85de08e4d6b3e3be5b97b'


def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_inputs(root):
    previous=set((root/'outputs/e004_e005_environment_lock.txt').read_text(encoding='utf-8').splitlines())
    current=set((root/'outputs/e006_e007_environment_lock.txt').read_text(encoding='utf-8').splitlines())
    assert current-previous=={'xgboost==3.4.1'} and not previous-current
    assert xgb.__version__=='3.4.1'
    path=root/'outputs/e006_e007_input_verification.json'
    if path.exists():
        record=read_json(path)
        for name,digest in record['protected_sha256'].items(): assert sha256(root/name)==digest,name
        for name,digest in record['input_sha256'].items(): assert sha256(name)==digest,name
        return record
    manifest=read_json(root/'outputs/full147_manifest.json')
    accepted=read_json(root/'outputs/baselines/input_verification.json')
    assert manifest['data_version']==accepted['data_version'] and len(manifest['feature_names'])==147
    inputs={v['path']:v['sha256'] for v in manifest['datasets']['train']['files'].values()}
    inputs[str(root/'outputs/baselines/cache/labels.npy')]=accepted['labels_sha256']
    inputs[str(root/'data/raw/训练集.csv')]=accepted['raw_sha256']
    for name,digest in inputs.items(): assert sha256(name)==digest,name
    assert (root/'evaluate.py').read_bytes()==(root/'reference/evaluate_official.py').read_bytes()
    assert sha256(root/'evaluate.py')==read_json(root/'outputs/official_evaluator_comparison.json')['script_sha256']
    # Snapshot only baseline outputs; all new stage outputs have the e006_e007 prefix.
    protected=list((root/'config').glob('*.yaml'))+[root/'evaluate.py',root/'reference/evaluate_official.py',root/'docs/FEATURE_SPEC_V1.md']
    protected += [p for p in (root/'outputs').glob('*') if p.is_file() and p.name!='experiment_log.csv' and not p.name.startswith('e006_e007')]
    for experiment in ['E000','E001','E002','E003','E004','E005']:
        protected += [p for p in (root/'outputs/baselines'/experiment).glob('*/*') if p.is_file()]
    # Preserve the entire previous training/evaluation implementation byte for byte.
    protected += [p for p in (root/'src/stock_prediction').glob('*.py') if p.name not in {'ranking.py','e006_e007.py','compare_models.py','restore_model_text_contract.py'}]
    record=dict(baseline_commit=BASELINE,input_sha256=inputs,data_version=manifest['data_version'],test_labels_used=False,
                protected_sha256={p.relative_to(root).as_posix():sha256(p) for p in protected},
                original_log_sha256=sha256(root/'outputs/experiment_log.csv'))
    save_json(path,record)
    print('Verified Full147, labels, frozen config, official evaluator and E000-E005 artifacts',flush=True)
    return record


def run_fold(root,experiment,fold_id):
    foldconfig,config=configs(root)
    model=make_model(config,experiment); model_id,target=EXPERIMENTS[experiment]
    fold=next(f for f in foldconfig['folds'] if f['id']==fold_id)
    manifest=read_json(root/'outputs/full147_manifest.json'); names=manifest['feature_names']
    source=manifest['datasets']['train']['files']
    matrix=np.load(source['matrix']['path'],mmap_mode='r',allow_pickle=False)
    keys=np.load(source['keys']['path'],mmap_mode='r',allow_pickle=False)
    labels=np.load(root/'outputs/baselines/cache/labels.npy',mmap_mode='r',allow_pickle=False)
    train,valid,split=split_fold(keys['trade_date'],labels,fold,foldconfig['purge_rule']['purge_trading_days'])
    prior=read_json(root/'outputs/baselines/E004'/fold_id/'result.json')
    environment=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,lightgbm=lgb.__version__)
    assert environment==prior['environment']
    assert split==prior['split'] and manifest['data_version']==prior['data_version']
    assert sha256(root/'outputs/full147_manifest.json')==prior['feature_manifest_sha256']
    assert {p.name:sha256(p) for p in (root/'config').glob('*.yaml')}==prior['config_sha256']
    out=root/'outputs/baselines'/experiment/fold_id; out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists(): raise FileExistsError(f'Completed {experiment}/{fold_id} exists')
    print(f'START {experiment}/{fold_id}: train={len(train):,}; valid={len(valid):,}; purge={split["purge_dates"]}',flush=True)
    started=time.perf_counter(); np.random.seed(config['seed'])
    order,fit_y,group,group_dates=training_plan(keys['trade_date'][train],labels[train],experiment)
    fit_indices=train[order]
    metadata=dict(name=target,definition=config['targets'][target],construction_scope='purged finite supervised training rows only',
                  training_indices_sha256=array_sha256(train),validation_indices_sha256=array_sha256(valid),
                  raw_training_labels_sha256=array_sha256(labels[train]),fit_indices_sha256=array_sha256(fit_indices),
                  values_sha256=array_sha256(fit_y),rows=len(fit_y),finite_rows=int(np.isfinite(fit_y).sum()),distribution=distribution(fit_y))
    for key in ['training_indices_sha256','validation_indices_sha256','raw_training_labels_sha256']:
        assert metadata[key]==prior['target_construction'][key]
    if experiment=='E006':
        assert metadata['values_sha256']==prior['target_construction']['values_sha256']
        np.testing.assert_array_equal(fit_indices,train)
    else:
        pd.DataFrame(dict(trade_date=group_dates,group_size=group)).to_csv(out/'ranking_groups.csv',index=False)
        metadata.update(group_sizes_sha256=array_sha256(group),group_dates_sha256=array_sha256(group_dates),
                        group_count=len(group),group_size_distribution=distribution(group),
                        relevance_histogram={str(i):int((fit_y==i).sum()) for i in range(10)},
                        ordering='stable trade_date ascending; original within-date row order')
    training=materialize(matrix,fit_indices,out/'training_work.npy')
    prep_seconds=time.perf_counter()-started
    fit_start=time.perf_counter()
    kwargs={} if group is None else {'group':group}
    # No validation data, sample weights, custom objective or early stopping.
    model.fit(pd.DataFrame(training,columns=names,copy=False),fit_y,**kwargs)
    fit_seconds=time.perf_counter()-fit_start
    assert iterations(model)==800
    print(f'FIT {experiment}/{fold_id}: {fit_seconds:.1f}s; predicting',flush=True)
    pred=predict_batches(model,matrix,valid,names)
    assert np.isfinite(pred).all()
    joblib.dump(dict(model=model,feature_names=names,preprocessor=None),out/'model.joblib')
    if experiment=='E006':
        (out/'model.ubj').write_bytes(model.get_booster().save_raw(raw_format='ubj'))
        save_json(out/'native_model_config.json',json.loads(model.get_booster().save_config()))
    else: save_lightgbm_text(model,out/'model.txt')
    training._mmap.close(); del training
    (out/'training_work.npy').unlink()
    p=pd.DataFrame(dict(ts_code=keys['ts_code'][valid],trade_date=keys['trade_date'][valid],pred=pred,
                        y_ret_1d=np.asarray(labels[valid]),flag_limit_up=matrix[valid,names.index('flag_limit_up')]))
    p.to_csv(out/'predictions.csv.gz',index=False,float_format='%.17g',compression={'method':'gzip','mtime':0})
    roundtrip=pd.read_csv(out/'predictions.csv.gz',float_precision='round_trip')
    np.testing.assert_array_equal(roundtrip.pred,pred)
    control=pd.read_csv(root/'outputs/baselines/E004'/fold_id/'predictions.csv.gz',float_precision='round_trip')
    pd.testing.assert_frame_equal(roundtrip.drop(columns='pred'),control.drop(columns='pred'))
    with tempfile.TemporaryDirectory(prefix='model_official_') as temp:
        official=replay(roundtrip,temp,load_official(root))
    saved=pd.read_csv(out/'predictions.csv.gz')
    daily,diag,selected,ties=official_daily_and_diagnostic(saved)
    for actual,expected in [(daily.rank_ic.mean(),official['ic_mean']),
                            (daily.rank_ic.std(ddof=1),official['ic_std']),
                            (daily.top_excess.mean()*252,official['annual_excess']),
                            (daily.top_return.mean()*252,official['top1_annual_ret']),
                            (diag['official_turnover'],official['mean_turnover'])]:
        np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)
    missing_top=selected.loc[selected.y_ret_1d.isna()].index.to_numpy()
    for label,idx in [('missing_y',saved.index[saved.y_ret_1d.isna()].to_numpy()),('missing_y_top',missing_top)]:
        diag[label+'_full147_finite_features']=distribution(np.isfinite(matrix[valid[idx]]).sum(axis=1))
    sample=np.unique(np.r_[np.linspace(0,len(valid)-1,200,dtype=int),missing_top[:20]])
    loaded=joblib.load(out/'model.joblib')['model']
    np.testing.assert_array_equal(predict_batches(loaded,matrix,valid[sample],names),pred[sample])
    daily.to_csv(out/'daily_metrics.csv',index=False,float_format='%.17g')
    ties.to_csv(out/'top_diagnostic_daily.csv',index=False)
    save_json(out/'missing_label_diagnostic.json',diag)
    metrics={METRIC_MAP[k]:v for k,v in official.items()}
    auxiliary=dict(rank_ic_std_ddof0=float(daily.rank_ic.std(ddof=0)),
                   top1_bottom1_annualized_spread=float(daily.auxiliary_floor_top_bottom_spread.mean()*252),
                   spread_status='auxiliary floor groups; not returned by organizer')
    resolved=model.get_params()
    if experiment=='E006':
        assert np.isnan(resolved['missing'])
        resolved['missing']='NaN'  # explicit JSON encoding, never fed back into fitting
    result=dict(experiment_id=experiment,fold_id=fold_id,run_id=f'{experiment}_{fold_id}_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}',
                timestamp=datetime.now(timezone.utc).isoformat(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
                data_version=manifest['data_version'],feature_spec_version=manifest['feature_spec_version'],protocol_version=config['experiment_protocol_version'],
                feature_set='Full147',n_features=147,target=target,model=model_id,seed=config['seed'],target_construction=metadata,
                configured_params=config['model_catalog'][model_id]['params'],resolved_params=resolved,actual_iterations=iterations(model),
                split=split,train_rows=len(train),valid_rows=len(valid),supervised_fit_rows=len(train),preprocessing=None,
                train_time_sec=fit_seconds,preprocessing_materialization_sec=prep_seconds,elapsed_seconds=time.perf_counter()-started,peak_memory_gb=peak_memory_gb(),
                evaluator_status='unmodified_organizer_script_validation_replay',official_script_sha256=sha256(root/'reference/evaluate_official.py'),
                official=official,metrics=metrics,auxiliary=auxiliary,missing_label_diagnostic=diag,
                validation=dict(e004_split_keys_labels_flags_equal=True,e006_target_and_fit_order_equal_e004=experiment=='E006',
                                group_dates_contiguous=experiment=='E007',saved_prediction_roundtrip=True,official_daily_aggregation_matches=True,model_reload_predictions_match=True),
                environment={**environment,'xgboost':xgb.__version__},
                config_sha256={p.name:sha256(p) for p in (root/'config').glob('*.yaml')},feature_manifest_sha256=sha256(root/'outputs/full147_manifest.json'),
                source_sha256={p.relative_to(root).as_posix():sha256(p) for p in (root/'src/stock_prediction').glob('*.py')},
                artifact_files={p.name:dict(sha256=sha256(p),bytes=p.stat().st_size) for p in out.iterdir() if p.is_file()})
    save_json(out/'result.json',result)
    fields=config['experiment_log']['required_columns']; row={k:result.get(k,'') for k in fields}
    row.update({k:split[k] for k in ['train_start','train_end','valid_start','valid_end']}); row.update(metrics)
    row.update(top1_bottom1_annualized_spread=auxiliary['top1_bottom1_annualized_spread'],
               params_json=json.dumps(result['resolved_params'],sort_keys=True),dropped_all_nan_features_json='[]',
               notes='Unmodified official evaluator; unchanged E004 inputs; E007 stable date groups; no tuning/postprocessing; diagnostic turnover separate')
    with (root/config['experiment_log']['path']).open('a',encoding='utf-8',newline='') as stream:
        csv.DictWriter(stream,fieldnames=fields).writerow(row)
    print(f'DONE {experiment}/{fold_id}: Score={official["final_score"]:.9f}; diagnostic turnover={diag["diagnostic_exclude_missing_y_turnover"]:.6f}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',choices=['E006','E007']); parser.add_argument('--fold',choices=['F1','F2','F3'])
    args=parser.parse_args()
    if bool(args.experiment)!=bool(args.fold): parser.error('Supply both --experiment and --fold')
    root=Path.cwd(); verify_inputs(root)
    if args.fold: run_fold(root,args.experiment,args.fold)
    else:
        for experiment in ['E006','E007']:
            for fold in ['F1','F2','F3']:
                subprocess.run([sys.executable,'-m','stock_prediction.e006_e007','--experiment',experiment,'--fold',fold],cwd=root,check=True)


if __name__=='__main__': main()
