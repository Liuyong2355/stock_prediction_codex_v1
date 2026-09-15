"""Read-only artifact replay and strict E003/E004/E005 target comparison."""
import json
from pathlib import Path
import subprocess
import tempfile

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .baselines import configs, predict_batches, save_json
from .build_basic40 import sha256
from .compare_official import load_official, replay
from .diagnose_e002 import top_sets
from .e004_e005 import verify_inputs
from .folds import split_fold
from .targets import TARGETS, array_sha256


def independent_target(dates, labels, target):
    """Independent full-vector oracle: NumPy date grouping, SciPy ranks."""
    result=np.empty(len(labels),dtype=np.float64)
    order=np.argsort(dates,kind='stable')
    boundaries=np.r_[0,np.flatnonzero(np.diff(dates[order]))+1,len(order)]
    for start,end in zip(boundaries[:-1],boundaries[1:]):
        idx=order[start:end]; values=labels[idx]
        if target=='rank': result[idx]=rankdata(values,method='average')/len(idx)-.5
        elif target=='relative': result[idx]=values-np.median(values)
        elif target=='raw': result[idx]=values
        else: raise ValueError(target)
    return result


def score_difference(a,b):
    result=dict(delta_score=b['final_score']-a['final_score'],
                ic_contribution=.4*(b['ic_mean']-a['ic_mean']),
                excess_contribution=.3*(b['annual_excess']-a['annual_excess']),
                turnover_contribution=.3*(a['mean_turnover']-b['mean_turnover']))
    np.testing.assert_allclose(result['delta_score'],sum(result[k] for k in
                               ['ic_contribution','excess_contribution','turnover_contribution']),rtol=0,atol=1e-14)
    return result


def native_parameters(path):
    text=path.read_text(encoding='utf-8')
    return text.split('parameters:\n',1)[1].split('end of parameters',1)[0]


def summarize(rows):
    metrics=['ic_mean','annual_excess','mean_turnover','final_score','diagnostic_turnover','missing_top_fraction']
    summary=[]
    for experiment in TARGETS:
        selected=[r for r in rows if r['experiment']==experiment]
        entry=dict(experiment=experiment,target=TARGETS[experiment])
        for metric in metrics:
            values=np.array([r[metric] for r in selected])
            worst=int(np.argmax(values) if metric in ['mean_turnover','diagnostic_turnover','missing_top_fraction'] else np.argmin(values))
            entry[metric]=dict(mean=float(values.mean()),worst=float(values[worst]),
                               worst_fold=selected[worst]['fold'],std_ddof0=float(values.std(ddof=0)))
        summary.append(entry)
    return summary


def compare(root):
    protected=verify_inputs(root)
    assert (root/'outputs/full147_e003_environment_lock.txt').read_text(encoding='utf-8').splitlines()==(root/'outputs/e004_e005_environment_lock.txt').read_text(encoding='utf-8').splitlines()
    foldconfig,config=configs(root)
    manifest=json.loads((root/'outputs/full147_manifest.json').read_text(encoding='utf-8'))
    source=manifest['datasets']['train']['files']; names=manifest['feature_names']
    matrix=np.load(source['matrix']['path'],mmap_mode='r',allow_pickle=False)
    keys=np.load(source['keys']['path'],mmap_mode='r',allow_pickle=False)
    labels=np.load(root/'outputs/baselines/cache/labels.npy',mmap_mode='r',allow_pickle=False)
    rows=[]; verification=[]
    for fold in foldconfig['folds']:
        fold_id=fold['id']
        train,valid,split=split_fold(keys['trade_date'],labels,fold,foldconfig['purge_rule']['purge_trading_days'])
        baseline=json.loads((root/'outputs/baselines/E003'/fold_id/'result.json').read_text(encoding='utf-8'))
        control=pd.read_csv(root/'outputs/baselines/E003'/fold_id/'predictions.csv.gz',float_precision='round_trip')
        assert split==baseline['split']
        np.testing.assert_array_equal(control.ts_code,keys['ts_code'][valid])
        np.testing.assert_array_equal(control.trade_date,keys['trade_date'][valid])
        np.testing.assert_array_equal(control.y_ret_1d,labels[valid])
        np.testing.assert_array_equal(control.flag_limit_up,matrix[valid,names.index('flag_limit_up')])
        for experiment,target in TARGETS.items():
            folder=root/'outputs/baselines'/experiment/fold_id
            result=json.loads((folder/'result.json').read_text(encoding='utf-8'))
            for name,meta in result['artifact_files'].items():
                assert sha256(folder/name)==meta['sha256'],(experiment,fold_id,name)
            for field in ['split','configured_params','resolved_params','environment','actual_iterations',
                          'preprocessing','feature_manifest_sha256','config_sha256','official_script_sha256',
                          'train_rows','valid_rows','supervised_fit_rows','data_version','feature_set','n_features',
                          'feature_spec_version','protocol_version','seed','model','evaluator_status']:
                assert result[field]==baseline[field],(experiment,fold_id,field)
            assert result['target']==target and result['actual_iterations']==800
            assert result['config_sha256']=={p.name:sha256(p) for p in (root/'config').glob('*.yaml')}
            assert result['feature_manifest_sha256']==sha256(root/'outputs/full147_manifest.json')
            if experiment!='E003':
                metadata=result['target_construction']
                assert metadata['definition']==config['targets'][target]
                assert metadata['training_indices_sha256']==array_sha256(train)
                assert metadata['validation_indices_sha256']==array_sha256(valid)
                assert metadata['raw_training_labels_sha256']==array_sha256(labels[train])
                oracle=independent_target(keys['trade_date'][train],labels[train],target)
                assert metadata['values_sha256']==array_sha256(oracle), (experiment,fold_id,'target oracle')
                assert metadata['rows']==metadata['finite_rows']==len(train)
                # Only training-path sources must be identical across all six runs.
                sources={name.replace('\\','/'):digest for name,digest in result['source_sha256'].items()}
                for name in ['e003.py','targets.py','e004_e005.py','baselines.py','folds.py',
                             'compare_official.py','diagnose_e002.py']:
                    path='src/stock_prediction/'+name
                    assert sources[path]==sha256(root/path),path
                del oracle
            p=pd.read_csv(folder/'predictions.csv.gz',float_precision='round_trip')
            pd.testing.assert_frame_equal(p.drop(columns='pred'),control.drop(columns='pred'))
            assert len(p)==len(valid) and not p.duplicated(['ts_code','trade_date']).any() and np.isfinite(p.pred).all()
            with tempfile.TemporaryDirectory(prefix='verify_targets_') as temp:
                official=replay(p,temp,load_official(root))
            for key,value in official.items():
                np.testing.assert_allclose(value,result['official'][key],rtol=0,atol=1e-14)
            # Reproduce every saved validation prediction using the persisted model.
            bundle=joblib.load(folder/'model.joblib'); model=bundle['model']
            assert bundle['feature_names']==names==model.feature_name_
            assert bundle['preprocessor'] is None and model.booster_.current_iteration()==800
            np.testing.assert_array_equal(predict_batches(model,matrix,valid),p.pred.to_numpy())
            # Verify the independently serialized native model on a fixed sample too.
            native=lgb.Booster(model_str=(folder/'model.txt').read_text(encoding='utf-8'))
            assert native_parameters(folder/'model.txt')==native_parameters(root/'outputs/baselines/E003'/fold_id/'model.txt')
            sample=np.linspace(0,len(valid)-1,257,dtype=int)
            np.testing.assert_allclose(native.predict(matrix[valid[sample]]),p.pred.iloc[sample],rtol=0,atol=1e-14)
            saved=pd.read_csv(folder/'predictions.csv.gz')
            selected,turnover,_=top_sets(saved)
            _,diagnostic_turnover,_=top_sets(saved,True)
            diag=result['missing_label_diagnostic']
            np.testing.assert_allclose(turnover,official['mean_turnover'],rtol=0,atol=1e-14)
            np.testing.assert_allclose(diagnostic_turnover,diag['diagnostic_exclude_missing_y_turnover'],rtol=0,atol=1e-14)
            assert len(selected)==diag['top_rows'] and int(selected.y_ret_1d.isna().sum())==diag['missing_top_rows']
            rows.append(dict(experiment=experiment,target=target,fold=fold_id,**official,
                             diagnostic_turnover=diagnostic_turnover,missing_top_fraction=diag['missing_share_of_top'],
                             missing_top_rows=diag['missing_top_rows'],top_rows=diag['top_rows']))
            verification.append(dict(experiment=experiment,fold=fold_id,artifact_hashes=True,
                                     official_saved_prediction_replay=True,all_validation_predictions_reproduced=True,
                                     native_model_sample_reproduced=True,strict_e003_control=True,
                                     native_parameters_including_thread_count_equal=True,
                                     independent_full_training_target='not applicable: frozen E003 raw' if experiment=='E003' else 'passed',
                                     train_rows=len(train),valid_rows=len(valid),purge_dates=split['purge_dates']))
            print(f'VERIFIED {experiment}/{fold_id}: Score={official["final_score"]:.9f}',flush=True)
    log_text=(root/'outputs/experiment_log.csv').read_text(encoding='utf-8')
    assert log_text.startswith(protected['original_log_text'])
    original_log=subprocess.check_output(['git','show',protected['baseline_commit']+':outputs/experiment_log.csv'],cwd=root)
    assert (root/'outputs/experiment_log.csv').read_bytes().startswith(original_log)
    log=pd.read_csv(root/'outputs/experiment_log.csv')
    assert len(log)==18 and not log.duplicated(['experiment_id','fold_id']).any()
    assert set(log.experiment_id)=={'E000','E001','E002','E003','E004','E005'}
    for row in rows:
        recorded=log.loc[(log.experiment_id==row['experiment'])&(log.fold_id==row['fold'])].iloc[0]
        np.testing.assert_allclose(recorded.final_score,row['final_score'],atol=1e-14,rtol=0)
    summary=summarize(rows); differences=[]
    for left,right in [('E003','E004'),('E003','E005'),('E004','E005')]:
        pair=[]
        for fold in ['F1','F2','F3']:
            a=next(r for r in rows if r['experiment']==left and r['fold']==fold)
            b=next(r for r in rows if r['experiment']==right and r['fold']==fold)
            pair.append(dict(comparison=f'{right} - {left}',fold=fold,**score_difference(a,b)))
        differences.extend(pair)
        differences.append(dict(comparison=f'{right} - {left}',fold='Mean',
                                **{key:float(np.mean([r[key] for r in pair])) for key in pair[0] if key not in ['comparison','fold']}))
    save_json(root/'outputs/e003_e004_e005_comparison.json',dict(
        scope='Frozen targets only; official validation scores, not hidden-test results',
        paired_rows=rows,summary=summary,score_difference_decomposition=differences,verification=verification,
        protected_prior_artifacts_unchanged=True,original_log_preserved=True,original_log_byte_prefix_matches_baseline=True,
        complete_dependency_lock_matches_e003=True,
        official_script_sha256=sha256(root/'evaluate.py'),stability_convention='population standard deviation across three folds (ddof=0)'))
    pd.DataFrame(rows).to_csv(root/'outputs/e003_e004_e005_comparison.csv',index=False,float_format='%.17g')
    write_report(root,rows,summary,differences)


def write_report(root,rows,summary,differences):
    lines=['# E003 / E004 / E005：冻结 target 严格对照','',
           '基线提交 eb9660a。三组均使用既有 Full147、F1–F3、一天 purge、相同监督训练行、147 列顺序、'
           'LightGBM 参数、seed=42、800 轮和相同依赖环境；唯一实验变量为训练 target。'
           '正式指标直接调用未修改的官方 evaluate.py，以下均为历史验证成绩。','',
           '## 三折结果','',
           '| 实验/target/折 | Rank IC | Top10% 超额年化 | 官方 Turnover | Official Score | 诊断 Turnover | Top 缺失占比 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['experiment']}/{r['target']}/{r['fold']} | {r['ic_mean']:.6f} | {r['annual_excess']:.6f} | {r['mean_turnover']:.6f} | {r['final_score']:.6f} | {r['diagnostic_turnover']:.6f} | {r['missing_top_fraction']:.2%} |")
    lines+=['','超额收益以小数表示（0.50 = 年化 50%），按日均 ×252，非复利。'
            'Top 缺失占比为官方换手 Top 中缺失原始标签的股票日数 / 全部 Top 股票日数。'
            '诊断 Turnover 在每个日期剔除缺失标签后重新选 Top，仅用于事后解释，未修改预测或官方评分。','',
            '## 均值、最差折与稳定性','',
            '以下标准差均为三折总体标准差 ddof=0；IC/收益/Score 的最差值取最小值，换手和缺失占比取最大值。','',
            '| 实验 | 指标 | 三折均值 | 最差值 | 最差折 | 三折标准差 |','|---|---|---:|---:|---|---:|']
    labels=dict(ic_mean='Rank IC',annual_excess='Top10% 超额年化',mean_turnover='官方 Turnover',final_score='Official Score',
                diagnostic_turnover='诊断 Turnover',missing_top_fraction='Top 缺失占比')
    for s in summary:
        for metric,label in labels.items():
            r=s[metric]
            lines.append(f"| {s['experiment']} | {label} | {r['mean']:.6f} | {r['worst']:.6f} | {r['worst_fold']} | {r['std_ddof0']:.6f} |")
    lines+=['','## 官方 Score 差值拆解','',
            'ΔScore = 0.4 × ΔIC + 0.3 × Δ超额收益 − 0.3 × ΔTurnover。正贡献表示提高 Score。','',
            '| 对比 | 折 | ΔScore | IC 贡献 | 收益贡献 | Turnover 贡献 |','|---|---|---:|---:|---:|---:|']
    for r in differences:
        lines.append(f"| {r['comparison']} | {r['fold']} | {r['delta_score']:+.6f} | {r['ic_contribution']:+.6f} | {r['excess_contribution']:+.6f} | {r['turnover_contribution']:+.6f} |")
    order=sorted(summary,key=lambda s:s['final_score']['mean'],reverse=True)
    lines+=['','## target 判断','',
            '按三折平均 Official Score 排序：'+' > '.join(f"{s['target']}（{s['experiment']}，{s['final_score']['mean']:.6f}）" for s in order)+'。','',
            f"均值领先的是 {order[0]['target']}；其最差折 Score={order[0]['final_score']['worst']:.6f}，"
            f"跨折标准差={order[0]['final_score']['std_ddof0']:.6f}。这是一组固定参数下的三折观察，不能推断所有参数或未来市场都保持此排序。"]
    winner=order[0]['experiment']
    if winner!='E003':
        winning_pair=[r for r in differences if r['comparison']==f'{winner} - E003']
        average=next(r for r in winning_pair if r['fold']=='Mean')
        best_fold=max((r for r in winning_pair if r['fold']!='Mean'),key=lambda r:r['delta_score'])
        lines+=['',f"按当前冻结参数下的官方综合分，优先选择 {order[0]['target']}。相对 Raw 的平均净提升 "
                f"{average['delta_score']:+.6f} 中，官方换手贡献 {average['turnover_contribution']:+.6f}"
                f"（约占净提升 {average['turnover_contribution']/average['delta_score']:.1%}）。"
                f"提升集中于 {best_fold['fold']}（ΔScore={best_fold['delta_score']:+.6f}），不是三折全面超过 Raw。"]
    for r in differences:
        if r['fold']=='Mean':
            largest=max(['ic_contribution','excess_contribution','turnover_contribution'],key=lambda k:abs(r[k]))
            label={'ic_contribution':'IC','excess_contribution':'收益','turnover_contribution':'turnover'}[largest]
            lines+=['',f"{r['comparison']}：平均 ΔScore={r['delta_score']:+.6f}；IC={r['ic_contribution']:+.6f}、"
                    f"收益={r['excess_contribution']:+.6f}、turnover={r['turnover_contribution']:+.6f}。最大绝对贡献来自 {label}。"]
    for metric,label in [('ic_mean','平均 Rank IC'),('annual_excess','平均 Top10% 超额年化')]:
        leader=max(summary,key=lambda s:s[metric]['mean'])
        lines+=['',f"{label} 最高的是 {leader['target']}（{leader[metric]['mean']:.6f}）。"]
    worst_leader=max(summary,key=lambda s:s['final_score']['worst'])
    stable=min(summary,key=lambda s:s['final_score']['std_ddof0'])
    lines+=['',f"最差折 Score 最高的是 {worst_leader['target']}（{worst_leader['final_score']['worst']:.6f}）；"
            f"跨折 Score 标准差最低的是 {stable['target']}（{stable['final_score']['std_ddof0']:.6f}）。"]
    for experiment in ['E004','E005']:
        paired=[r for r in differences if r['comparison']==f'{experiment} - E003' and r['fold']!='Mean']
        lines+=['',f"{experiment} 相对 Raw：Score 在 {sum(r['delta_score']>0 for r in paired)}/3 折提高，"
                f"IC 在 {sum(r['ic_contribution']>0 for r in paired)}/3 折提高，"
                f"超额收益在 {sum(r['excess_contribution']>0 for r in paired)}/3 折提高。"]
    lines+=['','### 缺失标签对换手解释的影响','',
            '下表的“缺失占比”是三个折占比的等权均值；每一折内部按股票日加权。'
            '官方与诊断 Top 使用不同候选集，差值不是可加到官方 Score 的修正项。','',
            '| target | 平均官方 Turnover | 平均诊断 Turnover | 诊断 − 官方 | 平均 Top 缺失占比 |','|---|---:|---:|---:|---:|']
    for s in summary:
        official=s['mean_turnover']['mean']; diagnostic=s['diagnostic_turnover']['mean']
        lines.append(f"| {s['target']} | {official:.6f} | {diagnostic:.6f} | {diagnostic-official:+.6f} | {s['missing_top_fraction']['mean']:.2%} |")
    raw=next(s for s in summary if s['experiment']=='E003')
    for s in summary:
        if s['experiment']=='E003': continue
        official_delta=s['mean_turnover']['mean']-raw['mean_turnover']['mean']
        diagnostic_delta=s['diagnostic_turnover']['mean']-raw['diagnostic_turnover']['mean']
        lines+=['',f"{s['target']} 相对 Raw：平均官方 turnover 变化 {official_delta:+.6f}，"
                f"诊断 turnover 变化 {diagnostic_delta:+.6f}；平均 Top 缺失占比从 {raw['missing_top_fraction']['mean']:.2%} "
                f"变为 {s['missing_top_fraction']['mean']:.2%}。"
                +('两种换手口径方向相反，官方换手分项不能单独当作有标签组合稳定性的判断。' if official_delta*diagnostic_delta<0 else '')]
    if winner!='E003':
        fold=best_fold['fold']
        a=next(r for r in rows if r['experiment']=='E003' and r['fold']==fold)
        b=next(r for r in rows if r['experiment']==winner and r['fold']==fold)
        lines+=['',f"重点看 {winner}/{fold}：Top 缺失占比由 Raw 的 {a['missing_top_fraction']:.2%} 变为 "
                f"{b['missing_top_fraction']:.2%}；官方 turnover {a['mean_turnover']:.6f} → {b['mean_turnover']:.6f}，"
                f"诊断 turnover {a['diagnostic_turnover']:.6f} → {b['diagnostic_turnover']:.6f}。"
                '官方换手的改善幅度与有标签诊断口径不同，保留该评分优势，但不能将其全部解释为可交易组合更稳定。']
    lines+=['','缺失样本占位与官方换手差异需要联合解读。仅凭 target 对照不能将全部换手差异因果归于缺失标签；'
            '诊断换手反映有标签候选集的 Top 变化，也不等同于计入交易限制和费用后的实际组合换手。']
    lines+=['','## 复现与无泄漏核验','',
            '- Rank = 同日有限训练标签平均百分位排名 −0.5；Relative = 原始训练收益 −同日中位数。先切折、purge、过滤非有限训练标签，再构造 target。',
            '- 三组使用原始 y_ret_1d 评价；模型只接收冻结 147 列，不传入验证标签、不使用 early stopping、不调整预测。',
            '- 六个新训练 target 全量通过 NumPy/SciPy 独立构造的逐字节指纹核验；训练/验证行索引、原始训练标签指纹一致。',
            '- 九组已保存预测直接通过官方脚本再次复算；九个模型重载后复现每一行验证预测，原生文本模型另做固定样本验证。',
            '- 原始数据、Full147 矩阵、键、标签缓存、冻结 YAML、官方脚本及 E000–E003 既有产物指纹保持一致；旧日志保留，仅追加六行。',
            '- 官方日 IC 标准差 ddof=1；冻结辅助 ddof=0 和 Top-Bottom spread 继续单独保存，不进入 Score。',
            '- 详细验证、逐日统计、模型/预测指纹、target 指纹和运行环境见对应 JSON、各折目录与 e004_e005_validation.md。',
            '- E006/E007、调参、融合、turnover 后处理均未执行。','']
    (root/'outputs/e003_e004_e005_comparison.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    compare(Path.cwd())
