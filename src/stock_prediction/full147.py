"""Frozen Full147: causal stock histories followed by finite same-date statistics."""
import numpy as np
import pandas as pd

from .basic40 import KEYS, SOURCES, EPSILON, _one_stock


def finite(series):
    return series.replace([np.inf, -np.inf], np.nan)


def paired_correlation(a, b, window):
    """Centered direct windows avoid cancellation; exactly w finite pairs required."""
    out = np.full(len(a), np.nan)
    if len(a) < window:
        return pd.Series(out, index=a.index)
    av = np.lib.stride_tricks.sliding_window_view(np.asarray(a, dtype=float), window)
    bv = np.lib.stride_tricks.sliding_window_view(np.asarray(b, dtype=float), window)
    valid = np.isfinite(av).all(axis=1) & np.isfinite(bv).all(axis=1)
    ac = av[valid] - av[valid].mean(axis=1, keepdims=True)
    bc = bv[valid] - bv[valid].mean(axis=1, keepdims=True)
    denom = np.sqrt((ac * ac).sum(axis=1) * (bc * bc).sum(axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        out[np.flatnonzero(valid) + window - 1] = (ac * bc).sum(axis=1) / denom
    return finite(pd.Series(out, index=a.index))


def ols_trend(close, window):
    slope, rsq = np.full(len(close), np.nan), np.full(len(close), np.nan)
    if len(close) >= window:
        views = np.lib.stride_tricks.sliding_window_view(close.to_numpy(float), window)
        valid = np.isfinite(views).all(axis=1)
        y = views[valid]
        yc = y - y.mean(axis=1, keepdims=True)
        x = np.arange(window, dtype=float) - (window - 1) / 2
        xx = np.dot(x, x)
        xy = yc @ x
        yy = (yc * yc).sum(axis=1)
        pos = np.flatnonzero(valid) + window - 1
        slope[pos] = xy / xx / (y[:, -1] + EPSILON)
        with np.errstate(divide="ignore", invalid="ignore"):
            rsq[pos] = xy * xy / (xx * yy)
        rsq[pos[yy == 0]] = np.nan
    return finite(pd.Series(slope, index=close.index)), finite(pd.Series(rsq, index=close.index))


def stock_features(raw, contract):
    """One complete stock history in ascending observed-row order, including test history."""
    x = raw[SOURCES].astype(float).replace([np.inf, -np.inf], np.nan)
    values = _one_stock(raw, contract['feature_sets']['Basic40']).to_dict('series')
    o, h, l, c = (x[k] for k in ['open', 'high', 'low', 'close'])
    ret = values['ret_1']
    def roll(s, w):
        return finite(s).rolling(w, min_periods=w)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        values['ret_5_minus_20'] = values['ret_5'] - values['ret_20']
        values['ret_20_minus_60'] = values['ret_20'] - values['ret_60']
        for w in [5, 10, 20, 60]:
            values[f'emadev_{w}'] = c / c.ewm(span=w, adjust=False, min_periods=w, ignore_na=False).mean() - 1
            values[f'slope_{w}'], values[f'rsq_{w}'] = ols_trend(c, w)
            lo, hi = roll(l, w).min(), roll(h, w).max()
            values[f'rsv_{w}'] = (c - lo) / (hi - lo + EPSILON)
            if w != 10:
                values[f'distmax_{w}'] = c / hi - 1
                values[f'distmin_{w}'] = c / lo - 1
        for a, b in [(5, 20), (10, 20), (20, 60)]:
            values[f'ma_{a}_over_{b}'] = roll(c, a).mean() / roll(c, b).mean() - 1
        hl = finite(np.log(h.where(h > 0) / l.where(l > 0)) ** 2)
        co = finite(np.log(c.where(c > 0) / o.where(o > 0)) ** 2)
        gk = finite(.5 * hl - (2 * np.log(2) - 1) * co)
        tr = np.maximum(np.maximum(h - l, (h - c.shift(1)).abs()), (l - c.shift(1)).abs())
        for w in [5, 10, 20, 60]:
            values[f'downvol_{w}'] = np.sqrt(roll(np.minimum(ret, 0) ** 2, w).mean())
            values[f'parkinson_{w}'] = np.sqrt(roll(hl, w).mean() / (4 * np.log(2)))
            values[f'gkvol_{w}'] = np.sqrt(roll(gk, w).mean().clip(lower=0))
            values[f'atr_{w}'] = roll(tr, w).mean() / (c + EPSILON)
        for col, ratio, change, logstd in [('vol', 'volratio', 'volchg', 'logvol_std'), ('amount', 'amtratio', 'amtchg', 'logamt_std')]:
            s = x[col]
            values[f'{ratio}_3'] = s / (roll(s, 3).mean() + EPSILON)
            logs = finite(np.log1p(s))
            for w in [1, 5, 20]:
                values[f'{change}_{w}'] = finite(logs - logs.shift(w))
            for w in [5, 20, 60]:
                values[f'{logstd}_{w}'] = roll(logs, w).std(ddof=0)
            for w in [5, 20]:
                suffix = 'vol' if col == 'vol' else 'amt'
                values[f'corr_ret_dlog{suffix}_{w}'] = paired_correlation(ret, values[f'{change}_1'], w)
        values['illiq_1'] = finite(np.log1p(1e8 * ret.abs() / x.amount.where(x.amount > 0)))
        for w in [5, 20, 60]:
            values[f'illiq_{w}'] = roll(values['illiq_1'], w).mean()
        for side in ['up', 'down']:
            flag = x[f'flag_limit_{side}']
            for w in [5, 20, 60]:
                values[f'limit_{side}_count_{w}'] = roll(flag, w).sum()
            positions = np.arange(len(x))
            last = np.maximum.accumulate(np.where(flag == 1, positions, -61))
            values[f'days_since_limit_{side}'] = pd.Series(np.minimum(positions - last, 61).astype(float), index=x.index).where(flag.notna())
    names = [f['name'] for f in contract['features'] if f['groupby'] == 'ts_code']
    assert set(values) == set(names)
    return pd.DataFrame(values, index=raw.index)[names].replace([np.inf, -np.inf], np.nan)


def cross_features(features, amount, contract):
    """Input has same-date source features; market amount rolling uses date series."""
    f = features.replace([np.inf, -np.inf], np.nan)
    dates = f.trade_date
    def aggregate(s, op):
        return s.groupby(dates, sort=True).transform(op)
    values = {'market_mean_ret_1': aggregate(f.ret_1, 'mean'),
              'market_breadth_up': aggregate((f.ret_1 > 0).astype(float).where(f.ret_1.notna()), 'mean')}
    for w in [1, 5, 20]:
        values[f'market_median_ret_{w}'] = aggregate(f[f'ret_{w}'], 'median')
    for w in [1, 5]:
        values[f'market_dispersion_ret_{w}'] = f[f'ret_{w}'].groupby(dates).transform('std', ddof=0)
    for side in ['up', 'down']:
        values[f'market_limit_{side}_ratio'] = aggregate(f[f'flag_limit_{side}'], 'mean')
    totals = finite(amount).groupby(dates, sort=True).sum(min_count=1)
    ratio = totals / (totals.rolling(20, min_periods=20).mean() + EPSILON)
    values['market_amount_ratio_20'] = dates.map(ratio)
    for definition in contract['features']:
        name = definition['name']
        if definition['category'] == 'relative':
            s = f[definition['depends_on'][0]]
            values[name] = s - aggregate(s, 'median')
        elif definition['category'] == 'cross_section_rank':
            s = f[definition['depends_on'][0]]
            values[name] = s.groupby(dates).rank(method='average', pct=True)
    return pd.DataFrame(values, index=f.index).replace([np.inf, -np.inf], np.nan)


def compute_full147(raw, contract):
    names = contract['feature_sets']['Full147']
    assert len(names) == len(set(names)) == 147
    frame = raw[KEYS + SOURCES].copy()
    if frame[KEYS].isna().any(axis=None) or frame.duplicated(KEYS).any():
        raise ValueError('Missing or duplicate stock/date keys')
    frame = frame.sort_values(KEYS, kind='stable').reset_index(drop=True)
    if frame.empty:
        return pd.DataFrame(columns=KEYS + names)
    stock = pd.concat([stock_features(g, contract) for _, g in frame.groupby('ts_code', sort=False)], axis=0)
    cross = cross_features(pd.concat([frame[['trade_date']], stock], axis=1), frame.amount, contract)
    return pd.concat([frame[KEYS], stock, cross], axis=1)[KEYS + names]
