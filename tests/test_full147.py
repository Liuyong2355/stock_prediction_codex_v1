from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from stock_prediction.basic40 import KEYS, SOURCES, load_contract, compute_basic40
from stock_prediction.full147 import compute_full147, stock_features, cross_features, paired_correlation, ols_trend
from test_basic40 import raw_stock

ROOT = Path(__file__).resolve().parents[1]
C = load_contract(ROOT / 'config/features_v1.yaml')
NAMES = C['feature_sets']['Full147']
STOCK_EXTRA = [f['name'] for f in C['features'] if f['groupby'] == 'ts_code' and f['name'] not in C['feature_sets']['Basic40']]


def reference(raw, t, name):
    """Independent scalar windows; numpy regression/correlation, no production helpers."""
    eps = 1e-12
    def a(col):
        return raw[col].to_numpy(float)
    c, o, h, l, v, amt = (a(k) for k in ['close', 'open', 'high', 'low', 'vol', 'amount'])
    r = np.r_[np.nan, c[1:] / c[:-1] - 1]
    def win(arr, w):
        z = arr[max(0, t-w+1):t+1]
        return z if len(z) == w and np.isfinite(z).all() else np.full(w, np.nan)
    w = int(name.split('_')[-1]) if name.split('_')[-1].isdigit() else None
    if name.startswith('ret_'):
        first, second = (5, 20) if name == 'ret_5_minus_20' else (20, 60)
        return c[t]/c[t-first] - c[t]/c[t-second] if t >= second else np.nan
    if name.startswith('emadev'):
        alpha = 2/(w+1); avg = np.nan; weight = 1.; count = 0
        for value in c[:t+1]:
            if np.isfinite(avg):
                weight *= 1-alpha
                if np.isfinite(value):
                    avg = (weight * avg + alpha * value)/(weight+alpha); weight=1.
            elif np.isfinite(value):
                avg=value
            count += int(np.isfinite(value))
        return c[t]/avg-1 if count >= w else np.nan
    if name.startswith('ma_'):
        first=int(name.split('_')[1]); return np.mean(win(c, first))/np.mean(win(c,w))-1
    if name.startswith(('slope_', 'rsq_')):
        y=win(c,w)
        if not np.isfinite(y).all(): return np.nan
        x=np.arange(w); design=np.column_stack([np.ones(w), x]); coeff=np.linalg.lstsq(design,y,rcond=None)[0]
        if name.startswith('slope'): return coeff[1]/(c[t]+eps)
        total=np.sum((y-y.mean())**2)
        return 1-np.sum((y-design@coeff)**2)/total if total>0 else np.nan
    if name.startswith('distmax'): return c[t]/np.max(win(h,w))-1
    if name.startswith('distmin'): return c[t]/np.min(win(l,w))-1
    if name.startswith('rsv'): return (c[t]-np.min(win(l,w)))/(np.max(win(h,w))-np.min(win(l,w))+eps)
    if name.startswith('downvol'): return np.sqrt(np.mean(win(np.minimum(r,0)**2,w)))
    if name.startswith('parkinson'): return np.sqrt(np.mean(win(np.log(h/l)**2,w))/(4*np.log(2)))
    if name.startswith('gkvol'): return np.sqrt(max(np.mean(win(.5*np.log(h/l)**2-(2*np.log(2)-1)*np.log(c/o)**2,w)),0))
    if name.startswith('atr'):
        prev=np.r_[np.nan,c[:-1]]; tr=np.max(np.column_stack([h-l,abs(h-prev),abs(l-prev)]),axis=1)
        return np.mean(win(tr,w))/(c[t]+eps)
    if name.startswith(('volratio','amtratio')):
        z=v if name.startswith('vol') else amt; return z[t]/(np.mean(win(z,w))+eps)
    if name.startswith(('volchg','amtchg')):
        z=v if name.startswith('vol') else amt; return np.log1p(z[t])-np.log1p(z[t-w]) if t>=w else np.nan
    if name.startswith(('logvol_std','logamt_std')):
        z=v if name.startswith('logvol') else amt; return np.std(win(np.log1p(z),w),ddof=0)
    if name.startswith('corr_'):
        z=v if 'dlogvol' in name else amt; change=np.r_[np.nan,np.diff(np.log1p(z))]
        ra, rb=win(r,w),win(change,w)
        return np.corrcoef(ra,rb)[0,1] if np.isfinite(ra).all() and np.isfinite(rb).all() else np.nan
    if name.startswith('illiq'):
        ill=np.log1p(1e8*np.abs(r)/np.where(amt>0,amt,np.nan))
        return ill[t] if w==1 else np.mean(win(ill,w))
    if name.startswith('limit_'):
        side=name.split('_')[1]; return np.sum(win(a('flag_limit_'+side),w))
    if name.startswith('days_since'):
        z=a('flag_limit_'+name.split('_')[-1]); event=np.flatnonzero(z[:t+1]==1)
        return min(t-event[-1],61) if len(event) else 61
    raise AssertionError(name)


@pytest.mark.parametrize('name', STOCK_EXTRA)
def test_frozen_extra_stock_formulas_independent_reference(name):
    raw=raw_stock(85); raw.loc[8,SOURCES[:6]]=np.nan
    raw.loc[[3,15], 'flag_limit_up']=1
    result=stock_features(raw,C)
    expected=[reference(raw,t,name) for t in range(len(raw))]
    np.testing.assert_allclose(result[name],expected,rtol=1e-8,atol=1e-11,equal_nan=True)


def test_full_membership_basic_exact_and_future_test_boundary():
    a=raw_stock(100); b=raw_stock(100,code='000002.SZ'); b[SOURCES[:6]]*=1.7
    raw=pd.concat([a,b],ignore_index=True)
    full=compute_full147(raw,C)
    assert list(full)==KEYS+NAMES and len(NAMES)==147
    pd.testing.assert_frame_equal(full[KEYS+C['feature_sets']['Basic40']],compute_basic40(raw,C))
    boundary=a.trade_date.iloc[74]
    prefix=compute_full147(raw.loc[raw.trade_date<=boundary],C)
    pd.testing.assert_frame_equal(full.loc[full.trade_date<=boundary].reset_index(drop=True),prefix)
    changed=raw.copy(); changed.loc[changed.trade_date>boundary,SOURCES[:6]]*=100
    changed['y_ret_1d']=1e9
    mutated=compute_full147(changed,C)
    pd.testing.assert_frame_equal(full.loc[full.trade_date<=boundary],mutated.loc[mutated.trade_date<=boundary])
    # Test first row consumes training history for EMA and market 20-day amount.
    isolated=compute_full147(raw.loc[raw.trade_date>boundary],C)
    assert isolated.iloc[0].emadev_60 != isolated.iloc[0].emadev_60
    assert np.isfinite(full.loc[full.trade_date==a.trade_date.iloc[75],'emadev_60']).all()
    assert np.isfinite(full.loc[full.trade_date==a.trade_date.iloc[75],'market_amount_ratio_20']).all()


def test_stock_independence_observed_gaps_and_market_dependence():
    a=raw_stock(80).drop(index=[4,7]).reset_index(drop=True)
    b=raw_stock(80,code='000002.SZ'); raw=pd.concat([a,b],ignore_index=True)
    first=compute_full147(raw,C)
    b[SOURCES[:6]]*=np.linspace(1,4,len(b))[:,None]
    second=compute_full147(pd.concat([a,b],ignore_index=True),C)
    cols=[f['name'] for f in C['features'] if f['groupby']=='ts_code']
    pd.testing.assert_frame_equal(first.loc[first.ts_code==a.ts_code.iloc[0],cols],second.loc[second.ts_code==a.ts_code.iloc[0],cols])
    assert first.iloc[4].ret_1==pytest.approx(a.close.iloc[4]/a.close.iloc[3]-1)
    assert not first.market_mean_ret_1.equals(second.market_mean_ret_1)


def test_finite_cross_sections_ties_empty_dates_and_amount_window():
    names=[f['name'] for f in C['features'] if f['groupby']=='ts_code']
    f=pd.DataFrame(0.,index=range(63),columns=names)
    f['trade_date']=np.repeat(np.arange(21),3)
    for n in ['ret_1','ret_5','ret_20','ret_60']:
        f[n]=np.tile([1.,1.,np.inf],21)
    f.loc[0:2,'ret_1']=[np.nan,np.inf,-np.inf]
    amount=pd.Series(np.tile([10.,20.,np.inf],21))
    out=cross_features(f,amount,C)
    assert out.loc[:2,'market_breadth_up'].isna().all()
    assert (out.loc[3:5,'market_breadth_up']==1).all()
    assert out.loc[3,'csr_ret_1']==.75 and out.loc[4,'csr_ret_1']==.75 and np.isnan(out.loc[5,'csr_ret_1'])
    assert (out.loc[3:5,'market_dispersion_ret_1']==0).all()
    assert out.loc[3,'relret_1']==0 and np.isnan(out.loc[5,'relret_1'])
    assert out.loc[:56,'market_amount_ratio_20'].isna().all()
    assert out.loc[57,'market_amount_ratio_20']==pytest.approx(1.)
    amount.iloc[:3]=np.nan
    out=cross_features(f,amount,C)
    assert np.isnan(out.loc[57,'market_amount_ratio_20']) and np.isfinite(out.loc[60,'market_amount_ratio_20'])


def test_complete_pairs_zero_variance_and_invalid_inputs():
    a=pd.Series([1.,2.,3.,4.,5.,6.]); b=pd.Series([1.,2.,np.inf,4.,5.,6.])
    corr=paired_correlation(a,b,3)
    assert corr.iloc[:5].isna().all() and corr.iloc[5]==pytest.approx(1.)
    assert paired_correlation(a,pd.Series(1.,index=a.index),3).isna().all()
    slope,rsq=ols_trend(pd.Series([2.]*8),5)
    assert rsq.isna().all() and (slope.iloc[4:]==0).all()
    raw=raw_stock(80); raw.loc[10,'low']=0.; raw.loc[20,'open']=-1.; raw.loc[30,'amount']=0.
    out=compute_full147(raw,C)
    assert out.loc[10:14,'parkinson_5'].isna().all()
    assert out.loc[20:24,'gkvol_5'].isna().all()
    assert out.loc[30:34,'illiq_5'].isna().all()
    assert not np.isinf(out[NAMES].to_numpy()).any()


def test_streaming_builder_matches_panel_and_preserves_test_history(tmp_path, monkeypatch):
    import json
    import shutil
    import stock_prediction.build_full147 as builder
    from stock_prediction.build_basic40 import sha256
    (tmp_path/'config').mkdir(); (tmp_path/'data/raw').mkdir(parents=True); (tmp_path/'outputs').mkdir()
    shutil.copyfile(ROOT/'config/features_v1.yaml',tmp_path/'config/features_v1.yaml')
    a=raw_stock(100); b=raw_stock(100,code='000002.SZ'); b['amount']*=2
    raw=pd.concat([a,b],ignore_index=True)
    expected=compute_full147(raw,C); basic=compute_basic40(raw,C)
    audit={'data_version':'fixture','datasets':{}}
    manifest={'data_version':'fixture','feature_names':C['feature_sets']['Basic40'],'datasets':{}}
    boundary=a.trade_date.iloc[74]
    for split,fn,mask in [('train','训练集.csv',raw.trade_date<=boundary),('test','测试集_X.csv',raw.trade_date>boundary)]:
        part=raw.loc[mask].copy()
        if split=='train': part['y_ret_1d']=0.
        path=tmp_path/'data/raw'/fn; part.to_csv(path,index=False)
        audit['datasets'][split]={'sha256':sha256(path),'flag_cross_field_consistency':{'both_limit_flags_equal_one':{'count':0}}}
        k=np.array(list(part[KEYS].itertuples(index=False,name=None)),dtype=[('ts_code','U16'),('trade_date','i8')])
        keypath=tmp_path/'outputs'/f'{split}_keys.npy'; np.save(keypath,k)
        matrixpath=tmp_path/'outputs'/f'{split}_basic.npy'; np.save(matrixpath,basic.loc[mask,C['feature_sets']['Basic40']].to_numpy())
        manifest['datasets'][split]={'files':{kind:{'path':str(p),'sha256':sha256(p),'size_bytes':p.stat().st_size} for kind,p in [('keys',keypath),('matrix',matrixpath)]}}
    (tmp_path/'outputs/basic40_manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    (tmp_path/'outputs/data_audit_summary.json').write_text(json.dumps(audit),encoding='utf-8')
    monkeypatch.setattr(builder.subprocess,'check_output',lambda *a,**k:'fixture')
    # CSV default parsing must match the accepted Basic40 artifact, including roundoff.
    from stock_prediction.audit import read_chunks
    raw_read=pd.concat([next(read_chunks(tmp_path/'data/raw/训练集.csv',True)),next(read_chunks(tmp_path/'data/raw/测试集_X.csv',False))],ignore_index=True).sort_values(KEYS).reset_index(drop=True)
    expected=compute_full147(raw_read,C); basic=compute_basic40(raw_read,C)
    for split,mask in [('train',basic.trade_date<=boundary),('test',basic.trade_date>boundary)]:
        p=tmp_path/'outputs'/f'{split}_basic.npy'; np.save(p,basic.loc[mask,C['feature_sets']['Basic40']].to_numpy())
        manifest['datasets'][split]['files']['matrix']['sha256']=sha256(p)
    (tmp_path/'outputs/basic40_manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    builder.build(tmp_path)
    for split,mask in [('train',expected.trade_date<=boundary),('test',expected.trade_date>boundary)]:
        actual=np.load(tmp_path/'outputs/features/full147'/f'{split}_full147.npy')
        np.testing.assert_allclose(actual,expected.loc[mask,NAMES],rtol=1e-12,atol=1e-12,equal_nan=True)
