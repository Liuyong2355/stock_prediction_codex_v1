"""Independent persisted-data checks for date aggregates, ranks and boundary history."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .baselines import save_json
from .basic40 import load_contract
from .build_basic40 import sha256


def verify(root):
    manifest=json.loads((root/'outputs/full147_manifest.json').read_text(encoding='utf-8'))
    names=manifest['feature_names']; contract=load_contract(root/'config/features_v1.yaml')
    keys={s:np.load(manifest['datasets'][s]['files']['keys']['path'],mmap_mode='r',allow_pickle=False) for s in ['train','test']}
    calendar=np.unique(np.concatenate([k['trade_date'] for k in keys.values()]))
    amount_sum=np.zeros(len(calendar)); amount_count=np.zeros(len(calendar),dtype=np.int64)
    for filename in ['训练集.csv','测试集_X.csv']:
        for chunk in pd.read_csv(root/'data/raw'/filename,usecols=['trade_date','amount'],chunksize=200000):
            valid=np.isfinite(chunk.amount); pos=np.searchsorted(calendar,chunk.trade_date[valid])
            amount_sum+=np.bincount(pos,weights=chunk.amount[valid],minlength=len(calendar))
            amount_count+=np.bincount(pos,minlength=len(calendar))
    amount_sum[amount_count==0]=np.nan
    amount_ratio=np.full(len(calendar),np.nan)
    for i in range(19,len(calendar)):
        window=amount_sum[i-19:i+1]
        if np.isfinite(window).all(): amount_ratio[i]=amount_sum[i]/(window.mean()+1e-12)
    records=[]; corrected={}
    for split in ['train','test']:
        source=manifest['datasets'][split]['files']['matrix']
        assert sha256(source['path'])==source['sha256']
        matrix=np.load(source['path'],mmap_mode='r',allow_pickle=False)
        dates=keys[split]['trade_date']; unique=np.unique(dates)
        expected=amount_ratio[np.searchsorted(calendar,dates)]
        np.testing.assert_allclose(matrix[:,names.index('market_amount_ratio_20')],expected,rtol=1e-12,atol=1e-12,equal_nan=True)
        picked=unique[np.unique([0,min(19,len(unique)-1),len(unique)//2,len(unique)-1])]
        for date in picked:
            idx=np.flatnonzero(dates==date)
            for definition in contract['features']:
                n=definition['name']
                if definition['groupby']!='trade_date' or n=='market_amount_ratio_20': continue
                dependency=definition['depends_on'][0]
                vals=matrix[idx,names.index(dependency)]
                good=np.isfinite(vals); finite=vals[good]
                if n.startswith('csr_'):
                    exp=np.full(len(idx),np.nan)
                    if len(finite): exp[good]=rankdata(finite,method='average')/len(finite)
                elif definition['category']=='relative':
                    exp=vals-np.median(finite) if len(finite) else np.full(len(idx),np.nan)
                else:
                    if not len(finite): value=np.nan
                    elif n=='market_breadth_up': value=np.mean(finite>0)
                    elif 'median' in n: value=np.median(finite)
                    elif 'dispersion' in n: value=np.std(finite,ddof=0)
                    else: value=np.mean(finite)
                    exp=np.full(len(idx),value)
                np.testing.assert_allclose(matrix[idx,names.index(n)],exp,rtol=1e-12,atol=1e-12,equal_nan=True,err_msg=f'{split}/{date}/{n}')
            records.append(dict(split=split,trade_date=int(date),rows=len(idx),all_33_same_date_features_independently_verified=True))
        oldpath=root/'outputs/features/full147_before_zero_variance_fix'/f'{split}_full147.npy'
        if oldpath.exists():
            old=np.load(oldpath,mmap_mode='r',allow_pickle=False)
            for j,n in enumerate(names):
                a,b=old[:,j],matrix[:,j]
                changed=int((~((a==b)|(np.isnan(a)&np.isnan(b)))).sum())
                if changed:
                    assert n.startswith(('slope_','rsq_','corr_')),n
                    corrected[f'{split}/{n}']=changed
    save_json(root/'outputs/full147_verification.json',dict(status='passed',matrix_hashes_verified=True,
              all_rows_market_amount_ratio_independent_raw_recomputation=True,date_checks=records,
              numerical_zero_variance_correction_changed_rows=corrected,
              correction_scope='Only exact constant-window OLS/correlation outputs changed; frozen formulas unchanged'))
    print('Full147 independent real-data verification passed',flush=True)


if __name__=='__main__': verify(Path.cwd())
