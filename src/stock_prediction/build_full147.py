"""Stream full stock histories, then complete date cross sections; audit all 147 columns."""
import json
import time
import subprocess
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path

import numpy as np
import pandas as pd

from .audit import read_chunks
from .basic40 import KEYS, load_contract, stock_batches
from .build_basic40 import sha256, audit_matrix
from .full147 import stock_features, cross_features
from .baselines import save_json


def build(root):
    start=time.perf_counter()
    contract=load_contract(root/'config/features_v1.yaml')
    names=contract['feature_sets']['Full147']
    assert len(names)==147 and len(set(names))==147
    stock_names=[f['name'] for f in contract['features'] if f['groupby']=='ts_code']
    cross_names=[f['name'] for f in contract['features'] if f['groupby']=='trade_date']
    dependencies=sorted({n for f in contract['features'] if f['groupby']=='trade_date' for n in f['depends_on']})
    basic=json.loads((root/'outputs/basic40_manifest.json').read_text(encoding='utf-8'))
    audit=json.loads((root/'outputs/data_audit_summary.json').read_text(encoding='utf-8'))
    assert basic['data_version']==audit['data_version']
    output=root/'outputs/features/full147'; output.mkdir(parents=True,exist_ok=True)
    matrices,keys,old,paths={}, {}, {}, {}
    for split,filename in [('train','训练集.csv'),('test','测试集_X.csv')]:
        assert sha256(root/'data/raw'/filename)==audit['datasets'][split]['sha256']
        for meta in basic['datasets'][split]['files'].values():
            assert sha256(meta['path'])==meta['sha256']
        keys[split]=np.load(basic['datasets'][split]['files']['keys']['path'],mmap_mode='r',allow_pickle=False)
        old[split]=np.load(basic['datasets'][split]['files']['matrix']['path'],mmap_mode='r',allow_pickle=False)
        paths[split]=output/f'{split}_full147.npy'
        if paths[split].exists() or Path(str(paths[split])+'.partial').exists():
            raise FileExistsError('Full147 artifact exists; do not overwrite')
        matrices[split]=np.lib.format.open_memmap(str(paths[split])+'.partial',mode='w+',dtype='float64',shape=(len(keys[split]),147),fortran_order=True)
    calendar=np.unique(np.concatenate([keys[s]['trade_date'] for s in keys]))
    totals=np.zeros(len(calendar)); amount_count=np.zeros(len(calendar),dtype=np.int64)
    offsets={'train':0,'test':0}; samples=[]
    iters=[stock_batches(read_chunks(root/'data/raw'/fn,tr)) for fn,tr in [('训练集.csv',True),('测试集_X.csv',False)]]
    for i,pair in enumerate(zip_longest(*iters),1):
        a,b=pair
        if a is None or b is None or a[0]!=b[0]: raise ValueError('Stock sets changed')
        code,train=a; _,test=b
        assert train.trade_date.max()<test.trade_date.min()
        raw=pd.concat([train,test],ignore_index=True)
        f=stock_features(raw,contract)
        for split,begin,end in [('train',0,len(train)),('test',len(train),len(raw))]:
            pos=offsets[split]; count=end-begin
            assert np.array_equal(keys[split]['ts_code'][pos:pos+count],raw.ts_code.iloc[begin:end].astype(str))
            assert np.array_equal(keys[split]['trade_date'][pos:pos+count],raw.trade_date.iloc[begin:end])
            np.testing.assert_array_equal(f[basic['feature_names']].iloc[begin:end].to_numpy(),old[split][pos:pos+count])
            for n in stock_names:
                matrices[split][pos:pos+count,names.index(n)]=f[n].iloc[begin:end]
            offsets[split]+=count
        positions=np.searchsorted(calendar,raw.trade_date)
        finite=np.isfinite(raw.amount)
        np.add.at(totals,positions[finite],raw.amount[finite].to_numpy())
        np.add.at(amount_count,positions[finite],1)
        if i==1:
            prefix=stock_features(train,contract)
            np.testing.assert_array_equal(prefix.to_numpy(),f.iloc[:len(train)].to_numpy())
            for position,reason in [(60,'61st observation'),(len(train),'first test row with training history')]:
                names_review=['emadev_5','slope_5','rsq_5','corr_ret_dlogvol_5','atr_5','illiq_5']
                samples.append(dict(reason=reason,ts_code=code,trade_date=int(raw.trade_date.iloc[position]),
                                    raw_history=json.loads(raw.iloc[max(0,position-5):position+1].to_json(orient='records')),
                                    features=f[names_review].iloc[position].to_dict(),
                                    ema_note='EMA uses full stock prefix; six displayed rows alone do not reproduce its state.'))
        if i%1000==0: print(f'Full147 stock stage: {i} stocks',flush=True)
    totals[amount_count==0]=np.nan
    series=pd.Series(totals,index=calendar)
    market_ratio=series/(series.rolling(20,min_periods=20).mean()+1e-12)
    for split in matrices:
        assert offsets[split]==len(keys[split])
        # Group only existing rows; no calendar rows are inserted.
        dates=keys[split]['trade_date']
        order=np.argsort(dates,kind='stable')
        sorted_dates=dates[order]
        cuts=np.r_[0,np.flatnonzero(sorted_dates[1:]!=sorted_dates[:-1])+1,len(order)]
        for left,right in zip(cuts[:-1],cuts[1:]):
            idx=order[left:right]; date=int(sorted_dates[left])
            frame=pd.DataFrame({n:matrices[split][idx,names.index(n)] for n in dependencies})
            frame['trade_date']=date
            # This one-date call handles all cross-sectional operations; the market
            # amount time series is computed above across the complete train/test calendar.
            cross=cross_features(frame,pd.Series(np.nan,index=frame.index),contract)
            cross['market_amount_ratio_20']=market_ratio.loc[date]
            for n in cross_names: matrices[split][idx,names.index(n)]=cross[n]
        matrices[split].flush(); matrices[split]._mmap.close()
        Path(str(paths[split])+'.partial').replace(paths[split])
        print(f'Full147 {split} cross sections complete; auditing persisted columns',flush=True)
    datasets={s:audit_matrix(paths[s],basic['datasets'][s]['files']['keys']['path'],names) for s in paths}
    for split,filename in [('train','训练集.csv'),('test','测试集_X.csv')]:
        assert sha256(root/'data/raw'/filename)==audit['datasets'][split]['sha256']
        assert datasets[split]['both_limit_flags_equal_one']==audit['datasets'][split]['flag_cross_field_consistency']['both_limit_flags_equal_one']['count']
    manifest=dict(feature_set='Full147',feature_spec_version=contract['feature_spec_version'],data_version=basic['data_version'],feature_names=names,
                  storage='float64 Fortran-order matrix; exact accepted Basic40 keys reused',datasets={})
    for split in paths:
        manifest['datasets'][split]=dict(rows=offsets[split],columns=147,files={
            'matrix':dict(path=str(paths[split].resolve()),sha256=sha256(paths[split]),size_bytes=paths[split].stat().st_size),
            'keys':basic['datasets'][split]['files']['keys']})
    summary=dict(feature_set='Full147',generated_at=datetime.now(timezone.utc).isoformat(),feature_spec_version=contract['feature_spec_version'],
                 data_version=basic['data_version'],feature_names=names,datasets=datasets,review_samples=samples,
                 config_sha256=sha256(root/'config/features_v1.yaml'),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
                 implementation_sha256={p.name:sha256(p) for p in [Path(__file__),Path(__file__).with_name('full147.py')]},
                 validation=dict(all_rows_keys_preserved=True,basic40_all_values_exactly_equal=True,raw_bytes_unchanged=True,
                                 first_stock_train_prefix_equal_with_or_without_test=True),elapsed_seconds=time.perf_counter()-start)
    save_json(root/'outputs/full147_manifest.json',manifest)
    save_json(root/'outputs/full147_audit_summary.json',summary)
    lines=['# Full147 全量特征审计','',f"数据版本：`{basic['data_version']}`；冻结特征规范：{contract['feature_spec_version']}。",'',
           '- 147 列全部生成；Basic40 子集全量逐值相等，键、行数和原始 flags 保留。',
           '- observed-row lag；普通 rolling 完整有限窗口；相关系数完整有限配对；EMA adjust=False/min_periods=span/ignore_na=False。',
           '- OLS 因变量零方差 R²=NaN；横截面仅当日 finite 值；空截面 amount 总额为 NaN，日期序列 rolling 不补日。',
           '- train/test 每股连接后因果计算；市场 amount 日期序列同样继承训练历史。NaN 不填充，极值不裁剪。',
           '- 全量精确 finite 分位数；逐列极值对应键及人工复核历史见同名 JSON。',
           '- 已知缺失样本仍保留；market/limit 特征可能使其获得更多有效输入，不据标签筛选。','']
    for split,dataset in datasets.items():
        lines += [f"## {split}: {dataset['rows']:,} × 147",'', '| 特征 | finite% | NaN% | min | p01 | median | p99 | max |','|---|---:|---:|---:|---:|---:|---:|---:|']
        for name,st in dataset['features'].items():
            vals=' | '.join('NaN' if st['quantiles'][q] is None else f"{st['quantiles'][q]:.8g}" for q in ['min','p01','p50','p99','max'])
            lines.append(f"| {name} | {st['finite_ratio']:.4%} | {st['nan_ratio']:.4%} | {vals} |")
    lines += ['', '## 核验', '', f"所有输出均为 finite 或 NaN，inf=0；涨跌停同时为 1 的记录原样保留：训练 {datasets['train']['both_limit_flags_equal_one']} 条，测试 {datasets['test']['both_limit_flags_equal_one']} 条。完整分位数/极值键/样例见 full147_audit_summary.json；文件指纹见 full147_manifest.json。", '']
    (root/'outputs/full147_audit_report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(f'Full147 complete in {time.perf_counter()-start:.1f}s',flush=True)


if __name__=='__main__':
    build(Path.cwd())
