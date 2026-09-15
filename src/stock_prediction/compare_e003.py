"""Verify persisted E003 results and report a paired official E002/E003 comparison."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .baselines import save_json
from .build_basic40 import sha256
from .compare_official import load_official, replay


def compare(root):
    verification=json.loads((root/'outputs/e003_input_verification.json').read_text(encoding='utf-8'))
    for name,digest in verification['protected_sha256'].items():
        assert sha256(root/name)==digest, name
    official_old=json.loads((root/'outputs/official_evaluator_comparison.json').read_text(encoding='utf-8'))
    diagnosis_old=json.loads((root/'outputs/e002_missing_label_diagnostic.json').read_text(encoding='utf-8'))
    rows=[]; verification_folds=[]
    for fold in ['F1','F2','F3']:
        folder=root/'outputs/baselines/E003'/fold
        result=json.loads((folder/'result.json').read_text(encoding='utf-8'))
        for name,meta in result['artifact_files'].items(): assert sha256(folder/name)==meta['sha256']
        p=pd.read_csv(folder/'predictions.csv.gz',float_precision='round_trip')
        with tempfile.TemporaryDirectory(prefix='verify_e003_') as directory:
            replayed=replay(p,directory,load_official(root))
        for key,value in replayed.items():
            assert np.isclose(value,result['official'][key],rtol=0,atol=1e-14,equal_nan=True),(fold,key)
        prior=next(r for r in official_old['folds'] if r['experiment']=='E002' and r['fold']==fold)
        prior_diag=next(r for r in diagnosis_old['folds'] if r['fold']==fold)
        for experiment,metrics,diag in [('E002',prior['official'],prior_diag),('E003',result['official'],result['missing_label_diagnostic'])]:
            rows.append(dict(experiment=experiment,fold=fold,**metrics,
                             diagnostic_turnover=diag['diagnostic_exclude_missing_y_turnover'],
                             missing_top_fraction=diag['missing_share_of_top'],missing_top_rows=diag['missing_top_rows'],
                             top_rows=diag['top_rows'],missing_top_duplicate_fraction=diag.get('missing_top_duplicate_fraction',diag.get('ties',{}).get('missing_top_duplicate_fraction')),
                             longest_missing_top_run=diag['persistence']['run_days'].get('max',0) if 'run_days' in diag['persistence'] else 0,
                             missing_top_finite_feature_distribution=diag.get('missing_y_top_full147_finite_features',diag.get('populations',{}).get('missing_y_in_top',{}).get('finite_features'))))
        verification_folds.append(dict(fold=fold,artifact_hashes='passed',official_saved_prediction_replay='passed',
                                       train_rows=result['train_rows'],valid_rows=result['valid_rows'],purge=result['split']['purge_dates'],
                                       actual_iterations=result['actual_iterations'],**result['validation']))
        print(f'Verified E003/{fold}',flush=True)
    summary=[]
    for experiment in ['E002','E003']:
        scores=np.array([r['final_score'] for r in rows if r['experiment']==experiment])
        summary.append(dict(experiment=experiment,MeanScore=float(scores.mean()),WorstScore=float(scores.min()),StdScore=float(scores.std(ddof=0))))
    differences=[]
    for fold in ['F1','F2','F3']:
        a=next(r for r in rows if r['experiment']=='E002' and r['fold']==fold)
        b=next(r for r in rows if r['experiment']=='E003' and r['fold']==fold)
        differences.append(dict(fold=fold,delta_score=b['final_score']-a['final_score'],
                                ic_contribution=.4*(b['ic_mean']-a['ic_mean']),
                                excess_contribution=.3*(b['annual_excess']-a['annual_excess']),
                                turnover_contribution=.3*(a['mean_turnover']-b['mean_turnover'])))
    log=pd.read_csv(root/'outputs/experiment_log.csv')
    assert len(log)==12 and not log.duplicated(['experiment_id','fold_id']).any()
    for r in rows:
        if r['experiment']=='E003':
            assert np.isclose(log.loc[(log.experiment_id=='E003')&(log.fold_id==r['fold']),'final_score'].iloc[0],r['final_score'],atol=1e-14,rtol=0)
    save_json(root/'outputs/e002_e003_comparison.json',dict(scope='Official validation scores, not hidden-test competition results',official_script_sha256=official_old['script_sha256'],
              paired_rows=rows,score_summary=summary,score_difference_decomposition=differences,verification=verification_folds,
              protected_e002_and_frozen_definitions_unchanged=True,notes=['diagnostic turnover filters unavailable y ex-post; not a score replacement',
              'official IC std ddof=1; frozen ddof=0 saved separately in each E003 auxiliary',
              'no E004-E007, tuning, ensembles, prediction filtering or missing-sample correction']))
    pd.DataFrame(rows).drop(columns=['missing_top_finite_feature_distribution']).to_csv(root/'outputs/e002_e003_comparison.csv',index=False)
    lines=['# E002 / E003 严格对照','',
           '同一原始数据、F1–F3、purge、监督训练行、验证键/标签/flag、LightGBM 参数和 800 轮。唯一实验变量是 Basic40 → Full147。评分均直接调用同一份未修改的官方 evaluate.py；不是隐藏测试集最终比赛成绩。','',
           '## 三折官方指标','',
           '| 实验/折 | Rank IC | Top10% 超额年化 | 官方 Turnover | Official Score | 诊断 Turnover（剔除 y 缺失） |',
           '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['experiment']}/{r['fold']} | {r['ic_mean']:.6f} | {r['annual_excess']:.6f} | {r['mean_turnover']:.6f} | {r['final_score']:.6f} | {r['diagnostic_turnover']:.6f} |")
    lines += ['', '| 实验 | 平均 Official Score | 最差折 | 三折标准差 |','|---|---:|---:|---:|']
    for r in summary: lines.append(f"| {r['experiment']} | {r['MeanScore']:.6f} | {r['WorstScore']:.6f} | {r['StdScore']:.6f} |")
    lines += ['', '## Score 变化拆解（E003 − E002）','', '| 折 | Score 差 | IC 项贡献 | 超额收益项贡献 | 换手项贡献 |','|---|---:|---:|---:|---:|']
    for r in differences: lines.append(f"| {r['fold']} | {r['delta_score']:+.6f} | {r['ic_contribution']:+.6f} | {r['excess_contribution']:+.6f} | {r['turnover_contribution']:+.6f} |")
    lines += ['', '## 缺失样本 Top10% 效应（原样保留）','',
              'Top 指官方换手 Top；比例按股票日数加权。诊断换手额外按 y 是否缺失重选 Top，使用事后信息，仅解释结果，未用于模型、预测修正或正式评分。','',
              '| 实验/折 | Top 中 y 缺失占比 | 缺失 Top 同日重复预测占比 | 缺失 Top 最长连续日数 | 缺失 Top 有效特征数中位数 |',
              '|---|---:|---:|---:|---:|']
    for r in rows:
        dup=r['missing_top_duplicate_fraction']; med=r['missing_top_finite_feature_distribution'].get('median')
        lines.append(f"| {r['experiment']}/{r['fold']} | {r['missing_top_fraction']:.2%} | {'NA' if dup is None else f'{dup:.2%}'} | {r['longest_missing_top_run']:.0f} | {'NA' if med is None else f'{med:.0f}'} |")
    lines += ['', 'Full147 对原始缺失行仍可产生 limit/market 等特征；这不是填充价格或标签。三折预测未做任何人为修正。低官方换手仍需结合缺失样本占位及诊断换手解释，不能直接视为可交易组合稳定性。', '',
              '## 验证与限制','',
              '- 三折保存预测再次通过原始脚本复算；结果、模型与预测指纹校验通过，800 轮及模型重新加载预测一致。',
              '- E002 历史文件、官方脚本、冻结特征/参数配置运行前后指纹一致；原实验日志保留，追加 E003 三行。',
              '- 官方 IC std 使用 ddof=1；用户冻结的 ddof=0 另存 auxiliary，不修改冻结 YAML。',
              '- 官方脚本未返回 Top-Bottom spread；附加统计采用与官方 Top 相同 floor 数量的两端组，明确标为 auxiliary，未进入 Official Score。',
              '- 不能仅凭一次特征包对照归因到某个特征族；本阶段未消融、调参、融合或运行 E004–E007。','']
    delta=summary[1]['MeanScore']-summary[0]['MeanScore']
    lines += ['## 简要结论','',f"Full147 的三折平均 Official Score 为 {summary[1]['MeanScore']:.6f}，较 Basic40 {'提高' if delta>0 else '降低'} {abs(delta):.6f}。各折收益/IC/换手变化见上表，缺失样本效应单独保留，不据此改变评分。",'']
    ic_gains=sum(r['ic_contribution']>0 for r in differences)
    excess_gains=sum(r['excess_contribution']>0 for r in differences)
    turnover_losses=sum(r['turnover_contribution']<0 for r in differences)
    lines += [f"Rank IC 在 {ic_gains}/3 折提高，Top10% 超额收益在 {excess_gains}/3 折提高；换手项在 {turnover_losses}/3 折降低 Score。预测信号指标与综合分数分开判断。",'']
    (root/'outputs/e002_e003_comparison.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__': compare(Path.cwd())
