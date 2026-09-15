"""Provisional PDF evaluator, NOT organizer-verified official scoring."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

POLICY = {
    "id": "pdf_provisional_v1", "status": "provisional", "official_parity_verified": False,
    "nonfinite_predictions": "Exclude from each metric; retain raw predictions in artifacts; report counts. Never fill.",
    "ic": "Finite pred/label pairs; at least 2 and both nonconstant; average Spearman ties; otherwise NaN.",
    "sort_ties": "pred descending, then ts_code ascending, stable and label-independent",
    "groups": "At least 10 eligible rows; numpy.array_split(sorted rows, 10); first=Top1, last=Bottom1.",
    "return_eligibility": "Finite pred and label, flag_limit_up != 1; market mean on this same eligible universe.",
    "turnover_eligibility": "Finite pred, flag_limit_up != 1; DO NOT filter missing labels or flag_limit_down.",
    "turnover_dates": "Within-fold consecutive observed dates only; first date NaN; invalid sets skip pair without bridging gaps.",
    "small_sample_dates": "Fewer than 10 eligible rows: returns or turnover set undefined for that metric, not zero.",
    "aggregation": "Mean over finite daily metric values; IC std ddof=0; positive fraction over defined IC days; no defined values -> NaN.",
    "icir_zero_std": "NaN", "annualization": "Daily arithmetic mean * 252, including auxiliary return metrics.",
    "empty_turnover_union": "NaN; never count empty-union days as zero turnover",
    "score": "0.40*rank_ic_mean + 0.30*annualized_top_excess_return + 0.30*(1-mean_turnover); no normalization",
}


def finite_mean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else np.nan


def groups(frame):
    if len(frame) < 10:
        return None
    ordered = frame.sort_values(["pred", "ts_code"], ascending=[False, True], kind="stable")
    indices = np.array_split(np.arange(len(ordered)), 10)
    return ordered.iloc[indices[0]], ordered.iloc[indices[-1]]


def evaluate(predictions):
    required = ["ts_code", "trade_date", "pred", "y_ret_1d", "flag_limit_up"]
    p = predictions[required].copy()
    if p[["ts_code", "trade_date"]].isna().any(axis=None) or p.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError("Evaluator requires unique nonmissing stock/date keys")
    if not p.flag_limit_up.isin([0, 1]).all():
        raise ValueError("Unexpected limit-up values; no automatic recoding")
    rows, previous = [], None
    for date, day in p.groupby("trade_date", sort=True):
        finite_pred = np.isfinite(day.pred)
        labeled = day.loc[finite_pred & np.isfinite(day.y_ret_1d)]
        ic = np.nan
        if len(labeled) >= 2 and labeled.pred.nunique() > 1 and labeled.y_ret_1d.nunique() > 1:
            ic = float(spearmanr(labeled.pred, labeled.y_ret_1d).statistic)
        returns = labeled.loc[labeled.flag_limit_up != 1]
        ret_groups = groups(returns)
        top_return = bottom_return = excess = np.nan
        if ret_groups is not None:
            top, bottom = ret_groups
            top_return, bottom_return = float(top.y_ret_1d.mean()), float(bottom.y_ret_1d.mean())
            excess = top_return - float(returns.y_ret_1d.mean())
        turnover_universe = day.loc[finite_pred & (day.flag_limit_up != 1)]
        turnover_groups = groups(turnover_universe)
        current = set(turnover_groups[0].ts_code) if turnover_groups is not None else None
        turnover = np.nan
        if current is not None and previous is not None and (current | previous):
            turnover = 1 - len(current & previous) / len(current | previous)
        previous = current
        rows.append({"trade_date": int(date), "rows": len(day), "finite_predictions": int(finite_pred.sum()),
                     "nonfinite_predictions": int((~finite_pred).sum()), "finite_labels": int(np.isfinite(day.y_ret_1d).sum()),
                     "ic_rows": len(labeled), "return_eligible_rows": len(returns),
                     "turnover_eligible_rows": len(turnover_universe),
                     "return_top_count": len(ret_groups[0]) if ret_groups else 0,
                     "return_bottom_count": len(ret_groups[1]) if ret_groups else 0,
                     "turnover_top_count": len(current) if current is not None else 0,
                     "rank_ic": ic, "top_return": top_return, "top_excess": excess,
                     "top_bottom_spread": top_return - bottom_return, "turnover": turnover})
    if not rows:
        raise ValueError("No validation rows")
    daily = pd.DataFrame(rows)
    valid_ic = daily.loc[np.isfinite(daily.rank_ic), "rank_ic"]
    ic_mean = finite_mean(valid_ic)
    ic_std = float(valid_ic.std(ddof=0)) if len(valid_ic) else np.nan
    excess = 252 * finite_mean(daily.top_excess)
    turnover = finite_mean(daily.turnover)
    metrics = {
        "rank_ic_mean": ic_mean, "rank_ic_std": ic_std,
        "icir": ic_mean / ic_std if ic_std > 0 else np.nan,
        "ic_positive_ratio": float((valid_ic > 0).mean()) if len(valid_ic) else np.nan,
        "annualized_top_excess_return": excess, "mean_turnover": turnover,
        "final_score": .4 * ic_mean + .3 * excess + .3 * (1 - turnover),
        "top1_annualized_absolute_return": 252 * finite_mean(daily.top_return),
        "top1_bottom1_annualized_spread": 252 * finite_mean(daily.top_bottom_spread),
    }
    coverage = {"validation_dates": len(daily), "defined_ic_dates": int(np.isfinite(daily.rank_ic).sum()),
                "defined_return_dates": int(np.isfinite(daily.top_excess).sum()),
                "defined_turnover_pairs": int(np.isfinite(daily.turnover).sum()),
                "validation_rows": len(p), "nonfinite_predictions": int((~np.isfinite(p.pred)).sum())}
    return {"evaluator_status": "provisional", "policy": POLICY, "metrics": metrics, "coverage": coverage}, daily
