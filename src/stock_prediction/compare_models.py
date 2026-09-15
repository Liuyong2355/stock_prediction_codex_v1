"""Verify frozen E006/E007 artifacts and compare E000-E007 using organizer scores."""
import json
from pathlib import Path
import subprocess
import tempfile

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata,spearmanr
import xgboost as xgb

from .baselines import configs,save_json
from .build_basic40 import sha256
from .compare_official import load_official,replay
from .compare_targets import score_difference
from .diagnose_e002 import top_sets
from .e006_e007 import BASELINE,read_json,verify_inputs
from .folds import split_fold
from .ranking import EXPERIMENTS,iterations,predict_batches
from .targets import array_sha256

METRICS=['ic_mean','annual_excess','mean_turnover','final_score','diagnostic_turnover','missing_top_fraction']


def independent_plan(dates,labels,experiment):
    order=np.argsort(dates,kind='stable')
    ordered_dates=dates[order]
    boundaries=np.r_[0,np.flatnonzero(np.diff(ordered_dates))+1,len(order)]
    result=np.empty(len(labels),dtype=np.float64 if experiment=='E006' else np.int32)
    for start,end in zip(boundaries[:-1],boundaries[1:]):
        idx=order[start:end]; pct=rankdata(labels[idx],method='average')/len(idx)
        result[idx]=pct-.5 if experiment=='E006' else np.minimum(np.floor(pct*10),9).astype(np.int32)
    if experiment=='E006': return np.arange(len(labels)),result,None,None
    return order,result[order],np.diff(boundaries).astype(np.int32),ordered_dates[boundaries[:-1]]


def summary_rows(rows):
    result=[]
    for experiment in sorted({r['experiment'] for r in rows}):
        selected=[r for r in rows if r['experiment']==experiment]
        assert len(selected)==3
        summary=dict(experiment=experiment,target=selected[0]['target'],model=selected[0]['model'])
        for metric in METRICS:
            values=np.array([np.nan if r[metric] is None else r[metric] for r in selected],dtype=float)
            complete=bool(np.isfinite(values).all())
            lower=metric in ['mean_turnover','diagnostic_turnover','missing_top_fraction']
            worst=(int(np.argmax(values)) if lower else int(np.argmin(values))) if complete else None
            summary[metric]=dict(mean=float(values.mean()) if complete else None,
                                 worst=float(values[worst]) if complete else None,
                                 worst_fold=selected[worst]['fold'] if complete else None,
                                 std_ddof0=float(values.std(ddof=0)) if complete else None,
                                 finite_folds=int(np.isfinite(values).sum()))
        result.append(summary)
    return result


def paired_differences(rows):
    result=[]
    for left,right in [('E004','E006'),('E004','E007'),('E005','E006'),('E005','E007'),('E006','E007')]:
        pair=[]
        for fold in ['F1','F2','F3']:
            a=next(r for r in rows if r['experiment']==left and r['fold']==fold)
            b=next(r for r in rows if r['experiment']==right and r['fold']==fold)
            pair.append(dict(comparison=f'{right} - {left}',fold=fold,**score_difference(a,b)))
        result.extend(pair)
        result.append(dict(comparison=f'{right} - {left}',fold='Mean',
                           **{k:float(np.mean([r[k] for r in pair])) for k in pair[0] if k not in ['comparison','fold']}))
    return result


def compare(root):
    verification=verify_inputs(root)
    foldconfig,config=configs(root)
    manifest=read_json(root/'outputs/full147_manifest.json'); names=manifest['feature_names']
    matrix=np.load(manifest['datasets']['train']['files']['matrix']['path'],mmap_mode='r',allow_pickle=False)
    keys=np.load(manifest['datasets']['train']['files']['keys']['path'],mmap_mode='r',allow_pickle=False)
    labels=np.load(root/'outputs/baselines/cache/labels.npy',mmap_mode='r',allow_pickle=False)
    old=read_json(root/'outputs/official_evaluator_comparison.json')
    rows=[]; checks=[]; agreement=[]
    for fold in foldconfig['folds']:
        fold_id=fold['id']
        train,valid,split=split_fold(keys['trade_date'],labels,fold,foldconfig['purge_rule']['purge_trading_days'])
        control=read_json(root/'outputs/baselines/E004'/fold_id/'result.json')
        p_control=pd.read_csv(root/'outputs/baselines/E004'/fold_id/'predictions.csv.gz',float_precision='round_trip')
        assert split==control['split']
        for experiment in [f'E{i:03d}' for i in range(8)]:
            folder=root/'outputs/baselines'/experiment/fold_id
            result=read_json(folder/'result.json')
            for name,item in result['artifact_files'].items(): assert sha256(folder/name)==item['sha256'],(experiment,fold_id,name)
            p=pd.read_csv(folder/'predictions.csv.gz',float_precision='round_trip')
            pd.testing.assert_frame_equal(p.drop(columns='pred'),p_control.drop(columns='pred'))
            np.testing.assert_array_equal(p.trade_date,keys['trade_date'][valid])
            np.testing.assert_array_equal(p.ts_code,keys['ts_code'][valid])
            np.testing.assert_array_equal(p.y_ret_1d,labels[valid])
            np.testing.assert_array_equal(p.flag_limit_up,matrix[valid,names.index('flag_limit_up')])
            assert len(p)==len(valid) and not p.duplicated(['ts_code','trade_date']).any()
            if experiment!='E000': assert np.isfinite(p.pred).all()
            with tempfile.TemporaryDirectory(prefix='all_models_official_') as temp:
                official=replay(p,temp,load_official(root))
            expected=result['official'] if experiment>='E003' else next(r['official'] for r in old['folds'] if r['experiment']==experiment and r['fold']==fold_id)
            for name,value in official.items():
                np.testing.assert_allclose(value,np.nan if expected[name] is None else expected[name],rtol=0,atol=1e-14,equal_nan=True)
            check=dict(experiment=experiment,fold=fold_id,artifact_hashes=True,official_replay=True,validation_keys_raw_labels_flags_equal=True)
            if experiment in EXPERIMENTS:
                for name in ['split','data_version','feature_spec_version','protocol_version','feature_set','n_features','seed',
                             'preprocessing','train_rows','valid_rows','supervised_fit_rows','feature_manifest_sha256','config_sha256','official_script_sha256','evaluator_status']:
                    assert result[name]==control[name],(experiment,fold_id,name)
                assert result['configured_params']==config['model_catalog'][EXPERIMENTS[experiment][0]]['params']
                assert result['target']==EXPERIMENTS[experiment][1]
                for name,value in control['environment'].items(): assert result['environment'][name]==value
                assert result['environment']['xgboost']=='3.4.1'
                for filename in ['ranking.py','e006_e007.py','e003.py','baselines.py','folds.py','targets.py','compare_official.py','diagnose_e002.py']:
                    path='src/stock_prediction/'+filename
                    assert result['source_sha256'][path]==sha256(root/path),path
                order,y,group,dates=independent_plan(keys['trade_date'][train],labels[train],experiment)
                metadata=result['target_construction']
                assert metadata['definition']==config['targets'][result['target']]
                assert metadata['training_indices_sha256']==array_sha256(train)
                assert metadata['validation_indices_sha256']==array_sha256(valid)
                assert metadata['raw_training_labels_sha256']==array_sha256(labels[train])
                assert metadata['fit_indices_sha256']==array_sha256(train[order])
                assert metadata['values_sha256']==array_sha256(y)
                assert metadata['rows']==metadata['finite_rows']==len(train)
                if experiment=='E006':
                    assert metadata['values_sha256']==control['target_construction']['values_sha256']
                    np.testing.assert_array_equal(order,np.arange(len(train)))
                else:
                    actual_groups=pd.read_csv(folder/'ranking_groups.csv')
                    np.testing.assert_array_equal(actual_groups.trade_date,dates)
                    np.testing.assert_array_equal(actual_groups.group_size,group)
                    assert metadata['group_sizes_sha256']==array_sha256(group)
                    assert metadata['group_dates_sha256']==array_sha256(dates)
                    assert len(group)==split['train_dates_after_purge'] and group.sum()==len(train)
                model=joblib.load(folder/'model.joblib')
                assert model['feature_names']==names and model['preprocessor'] is None
                model=model['model']
                assert isinstance(model,xgb.XGBRegressor if experiment=='E006' else lgb.LGBMRanker)
                assert iterations(model)==result['actual_iterations']==800 and model.n_features_in_==147
                for name,value in result['configured_params'].items(): assert model.get_params()[name]==value
                np.testing.assert_array_equal(predict_batches(model,matrix,valid,names),p.pred.to_numpy())
                sample=np.linspace(0,len(valid)-1,257,dtype=int)
                if experiment=='E006':
                    native=xgb.Booster(); native.load_model(bytearray((folder/'model.ubj').read_bytes()))
                    native_pred=native.predict(xgb.DMatrix(pd.DataFrame(matrix[valid[sample]],columns=names)))
                    native_config=read_json(folder/'native_model_config.json')
                    assert native_config['learner']['objective']['name']=='reg:squarederror'
                    assert native.feature_names==names
                else:
                    native=lgb.Booster(model_str=(folder/'model.txt').read_text(encoding='utf-8'))
                    native_pred=native.predict(matrix[valid[sample]])
                    assert native.feature_name()==names
                np.testing.assert_array_equal(native_pred,p.pred.iloc[sample].to_numpy())
                check.update(independent_full_labels_and_order=True,all_saved_predictions_reproduced=True,native_model_sample_reproduced=True,
                             train_rows=len(train),valid_rows=len(valid),purge_dates=split['purge_dates'],groups=None if group is None else len(group))
                daily=[]
                paired=p[['trade_date','pred','y_ret_1d']].copy(); paired['e004_pred']=p_control.pred.to_numpy()
                for date,day in paired.groupby('trade_date',sort=True):
                    eligible=day.loc[day.y_ret_1d.notna()]
                    daily.append(dict(trade_date=int(date),prediction_rank_correlation=float(spearmanr(eligible.pred,eligible.e004_pred).statistic)))
                pd.DataFrame(daily).to_csv(root/f'outputs/e006_e007_{experiment}_{fold_id}_prediction_agreement.csv',index=False)
                agreement.append(dict(experiment=experiment,fold=fold_id,e004_prediction_rank_correlation=float(np.mean([r['prediction_rank_correlation'] for r in daily]))))
                del model,y,order
            saved=pd.read_csv(folder/'predictions.csv.gz')
            selected,turnover,_=top_sets(saved)
            _,diagnostic_turnover,_=top_sets(saved,True)
            np.testing.assert_allclose(turnover,official['mean_turnover'],rtol=0,atol=1e-14)
            missing_share=float(selected.y_ret_1d.isna().mean())
            if experiment>='E003':
                diag=result['missing_label_diagnostic']
                assert diag['top_rows']==len(selected) and diag['missing_top_rows']==int(selected.y_ret_1d.isna().sum())
                np.testing.assert_allclose(diagnostic_turnover,diag['diagnostic_exclude_missing_y_turnover'],rtol=0,atol=1e-14)
            rows.append(dict(experiment=experiment,fold=fold_id,target=result['target'],model=result['model'],feature_set=result['feature_set'],**official,
                             diagnostic_turnover=diagnostic_turnover,missing_top_fraction=missing_share,top_rows=len(selected),missing_top_rows=int(selected.y_ret_1d.isna().sum())))
            checks.append(check)
            print(f'VERIFIED {experiment}/{fold_id}: official Score={official["final_score"]}',flush=True)
    original_log=subprocess.check_output(['git','show',BASELINE+':outputs/experiment_log.csv'],cwd=root)
    assert (root/'outputs/experiment_log.csv').read_bytes().startswith(original_log)
    assert sha256_bytes(original_log)==verification['original_log_sha256']
    log=pd.read_csv(root/'outputs/experiment_log.csv')
    assert len(log)==24 and not log.duplicated(['experiment_id','fold_id']).any()
    for r in rows:
        if r['experiment']>='E003':
            item=log.loc[(log.experiment_id==r['experiment'])&(log.fold_id==r['fold'])].iloc[0]
            np.testing.assert_allclose(item.final_score,r['final_score'],atol=1e-14,rtol=0)
    summary=summary_rows(rows); differences=paired_differences(rows)
    report=dict(scope='E006/E007 only trained; all E000-E007 primary metrics replay unmodified organizer evaluator',
                official_script_sha256=sha256(root/'evaluate.py'),paired_rows=rows,summary=summary,score_difference_decomposition=differences,
                prediction_agreement=agreement,verification=checks,protected_prior_artifacts_unchanged=True,original_log_byte_prefix_preserved=True,
                no_tuning_ablation_ensemble_or_postprocessing=True)
    save_json(root/'outputs/e006_e007_comparison.json',report)
    pd.DataFrame(rows).to_csv(root/'outputs/e006_e007_comparison.csv',index=False,float_format='%.17g')
    write_report(root,report)


def sha256_bytes(value):
    import hashlib
    return hashlib.sha256(value).hexdigest()


def fmt(value): return '未定义' if value is None or not np.isfinite(value) else f'{value:.6f}'


def write_report(root,report):
    rows=report['paired_rows']; summaries=report['summary']; differences=report['score_difference_decomposition']
    lines=['# E006 / E007：冻结模型对照与 Phase C 候选','',
           '基线 24e979d。本阶段只运行 E006（Full147 + XGBoost + Rank）和 E007（Full147 + LightGBM Ranker + rank-decile）。'
           '所有正式分数直接来自现有未修改的官方 evaluate.py；E000–E007 共 24 组已保存验证预测全部复算。','',
           '## 控制条件','',
           '- E006 与 E004 的 Full147 文件、147 列顺序、float64 输入、训练/验证行、原始标签、Rank 标签、purge、seed、800 轮、预测批次和评分流程一致；唯一变化是模型及其冻结参数包。',
           '- XGBoost 3.4.1 是本阶段按可用官方发行版预先固定的新增依赖；其余依赖与 E004 相同。内部数值表示/直方图和缺失值处理由官方库负责，不改变输入矩阵、不调库默认值。',
           '- E007 先 purge、过滤非有限训练标签，再按平均百分位构造 min(floor(rank_pct*10),9)，按 trade_date 稳定排序。每日期一个连续 group，组大小总和等于监督训练行数。验证行保持 E004 原顺序。',
           '- E007 对照 E004 同时涉及排序目标、relevance 和必要的训练行分组顺序；这不是仅改变一个损失函数的消融实验。',
           '- 均未传验证集拟合、early stopping、自定义训练算法、sample weight、特征消融、融合或预测后处理。','',
           '## 三折官方结果（E003–E007）','',
           '| 实验/折 | Rank IC | Top10% 超额年化 | 官方 Turnover | Official Score | 诊断 Turnover | Top 缺失占比 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['experiment']>='E003':
            lines.append(f"| {r['experiment']}/{r['fold']} | {fmt(r['ic_mean'])} | {fmt(r['annual_excess'])} | {fmt(r['mean_turnover'])} | {fmt(r['final_score'])} | {fmt(r['diagnostic_turnover'])} | {r['missing_top_fraction']:.2%} |")
    lines+=['','收益为日均 ×252 的小数年化值（0.50 表示 50%，非复利）。Top 缺失占比以官方换手 Top 的股票日数加权。'
            '诊断换手按原始标签是否缺失另行重选 Top，仅供事后解释，不进入官方 Score。','',
            '## 实验对应关系','',
            '| 实验 | 特征 | 官方库模型 / 基线 | target |','|---|---|---|---|']
    for s in summaries:
        r=next(r for r in rows if r['experiment']==s['experiment'])
        model_label={'identity':'pred = ret_1','ridge_v1':'sklearn.Ridge',
                     'lightgbm_reg_v1':'lightgbm.LGBMRegressor','xgboost_reg_v1':'xgboost.XGBRegressor',
                     'lightgbm_ranker_v1':'lightgbm.LGBMRanker'}[r['model']]
        lines.append(f"| {r['experiment']} | {r['feature_set']} | {model_label} | {r['target']} |")
    lines += ['',
            '## E000–E007 全局 Score 与信号概览','',
            '| 实验 | 平均 IC | 平均超额年化 | 平均官方换手 | 平均 Score | 最差 Score | Score 标准差 | 平均诊断换手 | 平均 Top 缺失占比 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        lines.append(f"| {s['experiment']} | {fmt(s['ic_mean']['mean'])} | {fmt(s['annual_excess']['mean'])} | {fmt(s['mean_turnover']['mean'])} | {fmt(s['final_score']['mean'])} | {fmt(s['final_score']['worst'])} | {fmt(s['final_score']['std_ddof0'])} | {fmt(s['diagnostic_turnover']['mean'])} | {s['missing_top_fraction']['mean']:.2%} |")
    lines+=['','E000 的官方 IC/Score 未定义，保留 NaN/null，不以 provisional 分数替换，也不对剩余折求均值。'
            'E000–E002 的历史 provisional 文件不改写，表内使用本次官方脚本复算结果。'
            '稳定性为三折总体标准差 ddof=0；官方日 IC 标准差 ddof=1 与冻结辅助 ddof=0 分开存储。','',
            '## 新模型的均值、最差折和稳定性','',
            '| 实验 | 指标 | 均值 | 最差值 | 最差折 | 三折标准差 |','|---|---|---:|---:|---|---:|']
    for s in summaries:
        if s['experiment'] not in EXPERIMENTS: continue
        for metric in METRICS:
            m=s[metric]
            lines.append(f"| {s['experiment']} | {metric} | {fmt(m['mean'])} | {fmt(m['worst'])} | {m['worst_fold']} | {fmt(m['std_ddof0'])} |")
    lines+=['','IC/收益/Score 的最差值取最小值；换手和缺失占比取最大值。','',
            '## Score 差值归因','',
            'ΔScore = 0.4×ΔIC + 0.3×Δ超额收益 −0.3×Δ官方 Turnover。','',
            '| 对照 | 折 | ΔScore | IC 贡献 | 收益贡献 | 换手贡献 |','|---|---|---:|---:|---:|---:|']
    for r in differences:
        lines.append(f"| {r['comparison']} | {r['fold']} | {r['delta_score']:+.6f} | {r['ic_contribution']:+.6f} | {r['excess_contribution']:+.6f} | {r['turnover_contribution']:+.6f} |")
    lines+=['','## Rank 信号的模型鲁棒性与 Ranker 判断','']
    lookup={s['experiment']:s for s in summaries}
    for experiment,label in [('E006','XGBoost Rank regression'),('E007','LightGBM Ranker')]:
        pair=[r for r in differences if r['comparison']==f'{experiment} - E004' and r['fold']!='Mean']
        s=lookup[experiment]
        positive=sum(r['ic_mean']>0 for r in rows if r['experiment']==experiment)
        avg=next(r for r in differences if r['comparison']==f'{experiment} - E004' and r['fold']=='Mean')
        lines += [f"{label}：Rank IC 为正 {positive}/3 折；相对 E004，IC 提高 {sum(r['ic_contribution']>0 for r in pair)}/3 折，"
                  f"收益提高 {sum(r['excess_contribution']>0 for r in pair)}/3 折，Score 提高 {sum(r['delta_score']>0 for r in pair)}/3 折。"
                  f"平均 ΔScore={avg['delta_score']:+.6f}（IC {avg['ic_contribution']:+.6f}、收益 {avg['excess_contribution']:+.6f}、换手 {avg['turnover_contribution']:+.6f}）。",'']
    x=lookup['E006']; control=lookup['E004']
    lines += [f"**XGBoost 验证了当前冻结条件下 Rank 信号的模型鲁棒性。** E006 平均 IC={fmt(x['ic_mean']['mean'])}，"
              f"与 E004 的 {fmt(control['ic_mean']['mean'])} 接近；三折收益和 Score 均提高，"
              f"最差折 Score 从 {fmt(control['final_score']['worst'])} 升至 {fmt(x['final_score']['worst'])}，"
              f"Score 标准差从 {fmt(control['final_score']['std_ddof0'])} 降至 {fmt(x['final_score']['std_ddof0'])}。"
              '平均得分提升主要来自收益，IC 贡献很小，官方换手平均略有拖累；这是小幅、跨折一致的改善，不是大幅领先。','',
              f"E006 平均 Top 缺失占比仅 {x['missing_top_fraction']['mean']:.2%}，低于 E004 的 "
              f"{control['missing_top_fraction']['mean']:.2%}，三折分别为 "
              + ' / '.join(f"{r['missing_top_fraction']:.2%}" for r in rows if r['experiment']=='E006')+'。'
              '因此其得分改善不依赖增加缺失标签 Top 占位；但有标签诊断换手没有整体改善，不能把收益优势解释成交易更稳定。','']
    ranker=lookup['E007']
    ranker_leads=ranker['final_score']['mean']>max(control['final_score']['mean'],x['final_score']['mean'])
    verdict='Ranker 的平均官方 Score 高于两种 Rank regression。' if ranker_leads else '当前冻结 Ranker 未优于 Rank regression。'
    lines += [f"**{verdict}** E007 平均 Score={fmt(ranker['final_score']['mean'])}，最差折 "
              f"{fmt(ranker['final_score']['worst'])}，标准差 {fmt(ranker['final_score']['std_ddof0'])}；"
              f"平均 IC={fmt(ranker['ic_mean']['mean'])}，平均超额年化={fmt(ranker['annual_excess']['mean'])}。"
              f"相对 E004 的逐折胜负与分项贡献见上表；相对 E006 的平均 ΔScore="
              f"{ranker['final_score']['mean']-x['final_score']['mean']:+.6f}。"
              '这里判断的是已冻结的 Ranker 配置，不能推广为所有 learning-to-rank 方法均无效。','']
    if ranker['ic_mean']['mean']<0 and ranker['annual_excess']['mean']<0:
        lines += [f"E007 的 IC 贡献为 {0.4*ranker['ic_mean']['mean']:+.6f}，收益贡献为 "
                  f"{0.3*ranker['annual_excess']['mean']:+.6f}，换手分项为 "
                  f"{0.3*(1-ranker['mean_turnover']['mean']):+.6f}。"
                  f"平均 Top 缺失占比 {ranker['missing_top_fraction']['mean']:.2%}；"
                  '综合分中的正向支撑来自低官方换手，并不代表有标签样本上的排序或收益有效。'
                  'group/relevance 已通过独立全量核验；不根据这些结果调参、翻转或修正预测。','']
    lines += ['| 对照 E004 的预测排名相关性（有原始标签样本） | F1 | F2 | F3 |','|---|---:|---:|---:|']
    for experiment in EXPERIMENTS:
        values=[r['e004_prediction_rank_correlation'] for r in report['prediction_agreement'] if r['experiment']==experiment]
        lines.append(f"| {experiment} 日相关系数均值 | "+' | '.join(fmt(v) for v in values)+' |')
    lines+=['','该相关性是同日期两模型预测的 Spearman 相关，仅辅助判断共同排序信号；不改变预测。'
            '两种模型/三段历史的结果只能说明当前冻结条件下的鲁棒性，不能证明 Rank target 对所有模型均最优。','',
            '## 缺失标签诊断','',
            '| 实验 | 平均官方换手 | 平均诊断换手 | 诊断 − 官方 | 平均 Top 缺失占比 |','|---|---:|---:|---:|---:|']
    for s in summaries:
        official=s['mean_turnover']['mean']; diagnostic=s['diagnostic_turnover']['mean']
        lines.append(f"| {s['experiment']} | {fmt(official)} | {fmt(diagnostic)} | {diagnostic-official:+.6f} | {s['missing_top_fraction']['mean']:.2%} |")
    lines+=['','同一模型的官方换手与诊断换手使用不同候选集。缺失样本长期占据 Top 可以降低官方换手，'
            '因此不能仅凭低官方换手断言可交易组合更稳定；诊断也不包含交易费用与完整交易约束。'
            '所有缺失样本的预测与官方评分原样保留。','',
            '## Phase C 候选建议（本阶段未执行）','',
            '| 实验 | 建议 | 依据与角色 |','|---|---|---|',
            '| E000 | 仅保留管线检查 | 官方 IC/Score 未定义，不进入模型择优。 |',
            '| E001 | 保留线性对照 | 官方综合分较低，作为模型复杂度的基准，不列为优先研究对象。 |',
            f"| E002 | 进入，作为主要评分基准 | Basic40 原始 target；平均 Score {fmt(lookup['E002']['final_score']['mean'])}。Top 缺失占比高，需保留诊断，不能将低官方换手直接解释成交易优势。 |",
            '| E003 | 保留 Raw 对照 | 用于辨认 target 与特征方案的作用；现有 Full147 综合分和最差折落后于 E005。 |',
            '| E004 | 保留 Rank 回归对照 | IC 强、Top 缺失占比较低；E006 三折综合分均略胜，可优先推进 E006，同时保留本模型作为同类信号参照。 |',
            f"| E005 | 优先进入 | 现有 LightGBM 回归三种 target 中，平均和最差折官方 Score 最好；平均 Score {fmt(lookup['E005']['final_score']['mean'])}。换手优势与缺失样本占位应联合解释。 |",
            '| E006 | 优先进入 | 跨模型验证 Rank 信号；较 E004 三折收益和 Score 均略升，Top 缺失占比很低，保留与 E005 不同的信号证据。 |',
            '| E007 | '+('进入候选进一步审查' if ranker_leads else '暂不进入优先候选')+' | '+
            ('平均官方 Score 领先，仍需联合最差折和缺失标签诊断判断。' if ranker_leads else '冻结配置的官方综合分未超过 Rank 回归；低换手不能单独作为保留理由。')+' |','',
            '建议 Phase C 优先候选为 E002、E005、E006；E002 保持官方评分基准角色，E003、E004、E001 保持对照角色。'
            '这是基于三段历史验证的候选建议，不是最终模型或参数选择，也未使用官方测试期标签。','',
            '## 验证与复现','',
            '- 24 组保存预测均直接调用官方脚本复算；历史分数与产物通过指纹核验。',
            '- 六个新模型保存后重载，逐值复现全部验证预测；原生 XGBoost UBJSON / LightGBM 文本模型另做固定样本验证。',
            '- 新模型六折的标签、拟合行序、group 日期/大小使用 NumPy/SciPy 独立实现全量重算并比对指纹。',
            '- 原实验日志逐字节前缀保留，仅追加 E006/E007 六行；共 24 行，无重复。',
            '- 详细执行与测试记录见 e006_e007_validation.md；复现命令见 docs/MODEL_COMPARISON.md。','',
            '| 实验/折 | 监督训练行数 | 验证行数 | purge 日期 | ranking groups |',
            '|---|---:|---:|---|---:|']
    for check in report['verification']:
        if check['experiment'] in EXPERIMENTS:
            groups='不适用' if check['groups'] is None else str(check['groups'])
            lines.append(f"| {check['experiment']}/{check['fold']} | {check['train_rows']:,} | {check['valid_rows']:,} | {check['purge_dates'][0]} | {groups} |")
    lines.append('')
    (root/'outputs/e006_e007_comparison.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__': compare(Path.cwd())
