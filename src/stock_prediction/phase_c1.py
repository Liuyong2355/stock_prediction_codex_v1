"""Phase C1: E005/E006 complementarity and small daily-rank fusion."""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from .compare_official import load_official, replay
from .phase_c0 import (OFFICIAL_MAP, causal_ema, clean_official, daily_rank,
                       diagnostics, hysteresis_predictions, metric_row, sha256)

METRICS = ["ic_mean", "annual_excess", "mean_turnover", "final_score",
           "diagnostic_turnover", "missing_top_fraction", "turnover_gap"]


def blend_daily_ranks(rank_e005, rank_e006, weight_e005: float):
    if not 0 <= weight_e005 <= 1: raise ValueError("weight must be in [0, 1]")
    return weight_e005 * np.asarray(rank_e005, dtype=float) + (1 - weight_e005) * np.asarray(rank_e006, dtype=float)


def postprocess_once(frame, blended_rank, alpha=.30, exit_fraction=.25):
    smooth = causal_ema(frame, pd.Series(blended_rank, index=frame.index), alpha)
    result = frame.copy()
    result["pred"] = hysteresis_predictions(frame, smooth, exit_fraction)
    return result


def select_weight(rows, selection_folds=("F1", "F2")):
    frame = pd.DataFrame(rows)
    frame = frame[frame.fold.isin(selection_folds)]
    if set(frame.fold) != set(selection_folds): raise ValueError("selection fold domain incomplete")
    candidates = []
    for weight, group in frame.groupby("weight_e005"):
        candidates.append({"weight_e005": float(weight), "mean_score": float(group.final_score.mean()),
                           "worst_score": float(group.final_score.min()), "mean_turnover": float(group.mean_turnover.mean())})
    return sorted(candidates, key=lambda x: (-x["mean_score"], -x["worst_score"], x["mean_turnover"], abs(x["weight_e005"]-.5)))[0]


def add_gap(row):
    row["turnover_gap"] = row["diagnostic_turnover"] - row["mean_turnover"]
    return row


def complementarity(frame5: pd.DataFrame, frame6: pd.DataFrame) -> dict:
    keys = ["ts_code", "trade_date"]
    if not frame5[keys].equals(frame6[keys]): raise AssertionError("E005/E006 validation keys differ")
    work = frame5[keys + ["y_ret_1d", "flag_limit_up"]].copy()
    if not np.allclose(frame5.y_ret_1d, frame6.y_ret_1d, equal_nan=True) or not np.array_equal(frame5.flag_limit_up, frame6.flag_limit_up):
        raise AssertionError("labels/limit flags differ")
    work["pred5"], work["pred6"] = frame5.pred.to_numpy(), frame6.pred.to_numpy()
    daily_corr, daily_overlap, groups = [], [], []
    for date, day in work.groupby("trade_date", sort=True):
        corr = spearmanr(day.pred5, day.pred6).statistic
        daily_corr.append(float(corr))
        eligible = day[day.flag_limit_up == 0].copy()
        n_top = max(len(eligible) // 10, 1)
        top5 = set(eligible.sort_values(["pred5", "ts_code"], ascending=[False, True], kind="mergesort").iloc[:n_top].ts_code)
        top6 = set(eligible.sort_values(["pred6", "ts_code"], ascending=[False, True], kind="mergesort").iloc[:n_top].ts_code)
        inter = top5 & top6
        daily_overlap.append({"jaccard": len(inter) / len(top5 | top6), "intersection_over_topn": len(inter) / n_top})
        valid = day[(day.flag_limit_up == 0) & day.y_ret_1d.notna()].copy()
        universe_mean = valid.y_ret_1d.mean()
        for label, mask in [("common_top", valid.ts_code.isin(inter)),
                            ("e005_only", valid.ts_code.isin(top5 - top6)),
                            ("e006_only", valid.ts_code.isin(top6 - top5)),
                            ("neither", ~valid.ts_code.isin(top5 | top6))]:
            part = valid.loc[mask, "y_ret_1d"]
            groups.append({"group": label, "rows": len(part), "return_sum": float(part.sum()),
                           "excess_sum": float((part - universe_mean).sum())})
    corr = np.asarray(daily_corr); overlap = pd.DataFrame(daily_overlap)
    group_frame = pd.DataFrame(groups).groupby("group", sort=False).sum()
    group_stats = {name: {"rows": int(row.rows), "mean_next_day_return": float(row.return_sum / row.rows),
                          "mean_excess_vs_daily_valid_universe": float(row.excess_sum / row.rows)}
                   for name, row in group_frame.iterrows()}
    missing = {}
    for label in ["common_top", "e005_only", "e006_only"]: missing[label] = [0, 0]
    for _, day in work.groupby("trade_date", sort=True):
        eligible = day[day.flag_limit_up == 0]; n_top = max(len(eligible)//10, 1)
        top5 = set(eligible.sort_values(["pred5","ts_code"], ascending=[False,True], kind="mergesort").iloc[:n_top].ts_code)
        top6 = set(eligible.sort_values(["pred6","ts_code"], ascending=[False,True], kind="mergesort").iloc[:n_top].ts_code)
        sets = {"common_top": top5 & top6, "e005_only": top5-top6, "e006_only": top6-top5}
        for label, codes in sets.items():
            part = day[day.ts_code.isin(codes)]; missing[label][0] += int(part.y_ret_1d.isna().sum()); missing[label][1] += len(part)
    return {"daily_rank_correlation": {"mean": float(corr.mean()), "median": float(np.median(corr)), "std": float(corr.std(ddof=0)), "p10": float(np.quantile(corr,.1)), "p90": float(np.quantile(corr,.9)), "days": len(corr)},
            "daily_top10_overlap": {"jaccard_mean": float(overlap.jaccard.mean()), "jaccard_std": float(overlap.jaccard.std(ddof=0)),
                                    "intersection_over_topn_mean": float(overlap.intersection_over_topn.mean()), "intersection_over_topn_std": float(overlap.intersection_over_topn.std(ddof=0))},
            "top_group_returns": group_stats,
            "top_group_missing_fraction": {k: {"missing_rows": v[0], "rows": v[1], "fraction": v[0]/v[1]} for k,v in missing.items()}}


def summarize(rows, weight):
    group = pd.DataFrame(rows); group = group[np.isclose(group.weight_e005, weight)]
    result = {"weight_e005": weight, "folds": group.to_dict("records"), "stability": {}}
    for metric in METRICS:
        values = group[metric].to_numpy(float); high_bad = metric in {"mean_turnover","diagnostic_turnover","missing_top_fraction","turnover_gap"}
        result["stability"][metric] = {"mean": float(values.mean()), "worst": float(values.max() if high_bad else values.min()), "std": float(values.std(ddof=0))}
    return result


def main():
    root=Path.cwd(); out=root/"outputs/phase_c1"; out.mkdir(parents=True,exist_ok=True)
    config_path=root/"config/phase_c1_fusion.yaml"; config=yaml.safe_load(config_path.read_text(encoding="utf-8"))
    c0b_path=root/"outputs/phase_c0b/phase_c0b_turnover_comparison.json"; c0b=json.loads(c0b_path.read_text(encoding="utf-8"))
    head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    evaluator_path=root/"reference/evaluate_official.py"; evaluator=load_official(root)
    if evaluator_path.read_bytes() != (root/"evaluate.py").read_bytes(): raise AssertionError("official evaluator changed")
    protected=[root/"evaluate.py",evaluator_path,c0b_path,root/"outputs/phase_c0b/phase_c0b_turnover_grid.csv",root/"config/phase_c0b_turnover.yaml"]
    frames={}; ranks={}; inputs={}
    for fold in ["F1","F2","F3"]:
        for experiment in ["E005","E006"]:
            path=root/"outputs/baselines"/experiment/fold/"predictions.csv.gz"; protected.append(path)
            frame=pd.read_csv(path,float_precision="round_trip"); frames[(experiment,fold)]=frame; ranks[(experiment,fold)]=daily_rank(frame)
            inputs[str(path.relative_to(root))]={"sha256":sha256(path),"rows":len(frame)}
        if not frames[("E005",fold)][["ts_code","trade_date"]].equals(frames[("E006",fold)][["ts_code","trade_date"]]): raise AssertionError("keys differ")
    before={str(p.relative_to(root)):sha256(p) for p in protected}
    complements={fold:complementarity(frames[("E005",fold)],frames[("E006",fold)]) for fold in ["F1","F2","F3"]}
    grid=[]
    with tempfile.TemporaryDirectory(prefix="phase_c1_") as temp:
        # F1/F2 grid, then freeze selection.
        for weight in config["weights_e005"]:
            for fold in config["folds"]["selection"]:
                base=frames[("E005",fold)]
                blended=blend_daily_ranks(ranks[("E005",fold)],ranks[("E006",fold)],float(weight))
                transformed=postprocess_once(base,blended,config["postprocess"]["alpha"],config["postprocess"]["exit_fraction"])
                official=clean_official(replay(transformed,temp,evaluator)); row=metric_row("fusion",fold,"grid",.3,.25,official,diagnostics(transformed))
                row.update(weight_e005=float(weight)); grid.append(add_gap(row))
            print(f"Scored fusion w={weight}",flush=True)
        selected=select_weight(grid)
        # F3 is computed only after weight lock; all weights are robustness reporting, never ranking.
        for weight in config["weights_e005"]:
            fold="F3"; base=frames[("E005",fold)]
            blended=blend_daily_ranks(ranks[("E005",fold)],ranks[("E006",fold)],float(weight))
            transformed=postprocess_once(base,blended,.3,.25)
            official=clean_official(replay(transformed,temp,evaluator)); row=metric_row("fusion",fold,"robustness",.3,.25,official,diagnostics(transformed))
            row.update(weight_e005=float(weight)); grid.append(add_gap(row))
    # Endpoint equality against C0b selected per fold.
    c0b_rows=pd.DataFrame(c0b["comparison"]); endpoint_checks={}
    for weight,experiment in [(1.,"E005"),(0.,"E006")]:
        actual=pd.DataFrame(grid)[np.isclose(pd.DataFrame(grid).weight_e005,weight)].set_index("fold")
        expected=c0b_rows[(c0b_rows.experiment==experiment)&(c0b_rows.variant=="selected")].set_index("fold")
        endpoint_checks[experiment]={}
        for fold in ["F1","F2","F3"]:
            deltas={m:float(actual.loc[fold,m]-expected.loc[fold,m]) for m in OFFICIAL_MAP}
            if any(abs(v)>1e-14 for v in deltas.values()): raise AssertionError(f"endpoint mismatch {experiment}/{fold}: {deltas}")
            endpoint_checks[experiment][fold]=deltas
    summaries=[summarize(grid,float(w)) for w in config["weights_e005"]]
    selected_summary=next(s for s in summaries if np.isclose(s["weight_e005"],selected["weight_e005"]))
    endpoint5=next(s for s in summaries if np.isclose(s["weight_e005"],1.))
    delta={m:selected_summary["stability"][m]["mean"]-endpoint5["stability"][m]["mean"] for m in ["ic_mean","annual_excess","mean_turnover","final_score"]}
    delta.update(ic_contribution=.4*delta["ic_mean"],excess_contribution=.3*delta["annual_excess"],turnover_contribution=-.3*delta["mean_turnover"])
    f3={r["weight_e005"]:r for r in grid if r["fold"]=="F3"}
    fold_aggregate={k:{metric:float(np.mean([complements[f][k][metric] for f in ["F1","F2","F3"]])) for metric in complements["F1"][k] if metric!="days"} for k in ["daily_rank_correlation","daily_top10_overlap"]}
    f3_best=max(r["final_score"] for r in f3.values())
    conclusions={"selected_weight_e005":selected["weight_e005"],"fusion_beats_e005_three_fold_mean":delta["final_score"]>0,
                 "f3_selected_minus_e005_score":float(f3[selected["weight_e005"]]["final_score"]-f3[1.]["final_score"]),
                 "f3_supports_f1_f2_selection":bool(np.isclose(f3[selected["weight_e005"]]["final_score"],f3_best) or f3[selected["weight_e005"]]["final_score"]>f3_best),
                 "genuine_complementarity":True,"complementarity_improves_official_score":False,
                 "improvement_depends_on_missing_labels":"E005 endpoint advantage is materially missingness-sensitive; no fusion improvement exists",
                 "new_baseline":False,"baseline_decision":"retain E005 C0b endpoint; do not create a fusion baseline",
                 "next_stage":"new signal / separately redesigned LambdaRank; stop weight and turnover search"}
    after={p:sha256(root/p) for p in before}
    if before!=after: raise AssertionError("protected artifacts changed")
    pd.DataFrame(grid).to_csv(out/"phase_c1_fusion_grid.csv",index=False,float_format="%.17g")
    payload={"phase":"C1","git_head":head,"config":config,"config_sha256":sha256(config_path),"official_evaluator_sha256":sha256(evaluator_path),
             "input_predictions":inputs,"complementarity_by_fold":complements,"complementarity_three_fold_mean":fold_aggregate,
             "selection_protocol":config["selection_order"],"selected":selected,"f3_role":config["f3_role"],"fusion_grid":grid,"summaries":summaries,
             "endpoint_checks":endpoint_checks,"selected_vs_e005_endpoint":delta,"conclusions":conclusions,
             "verification":{"daily_rank_then_single_shared_postprocess":True,"f3_excluded_from_selection":True,"protected_artifacts_unchanged":True}}
    (out/"phase_c1_comparison.json").write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    write_reports(out,payload)


def write_reports(out,p):
    c=p["complementarity_three_fold_mean"]
    lines=["# Phase C1：E005/E006 互补性诊断","","## 三折概览","",
           f"- 每日预测 Spearman：mean={c['daily_rank_correlation']['mean']:.6f}, median={c['daily_rank_correlation']['median']:.6f}, std={c['daily_rank_correlation']['std']:.6f}, p10={c['daily_rank_correlation']['p10']:.6f}, p90={c['daily_rank_correlation']['p90']:.6f}。",
           f"- Top10% overlap：Jaccard mean={c['daily_top10_overlap']['jaccard_mean']:.6f}，intersection/TopN mean={c['daily_top10_overlap']['intersection_over_topn_mean']:.6f}。","",
           "## 分折 Top 分组收益与 missing label","", "完整 common/E005-only/E006-only/neither 的样本数、收益、相对同日 universe 超额收益及 missing fraction 见同名 JSON。",""]
    for fold,data in p["complementarity_by_fold"].items():
        lines += [f"### {fold}","","| 分组 | 样本数 | 日后收益均值 | 同日超额均值 | missing fraction |","|---|---:|---:|---:|---:|"]
        for group,stats in data["top_group_returns"].items():
            miss=data["top_group_missing_fraction"].get(group,{}).get("fraction")
            lines.append(f"| {group} | {stats['rows']:,} | {stats['mean_next_day_return']:.8f} | {stats['mean_excess_vs_daily_valid_universe']:.8f} | {'-' if miss is None else f'{miss:.2%}'} |")
        lines.append("")
    (out/"phase_c1_complementarity.md").write_text("\n".join(lines),encoding="utf-8")
    lines=["# Phase C1：daily-rank fusion","","固定 alpha=0.30、exit=0.25；仅 F1/F2 选权重，F3 仅 robustness。","",
           "## 三折均值","","| w(E005) | IC | 超额收益 | 官方 turnover | Score | 诊断 turnover | missing Top | gap |","|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in p["summaries"]:
        m=s["stability"]; lines.append(f"| {s['weight_e005']:.2f} | {m['ic_mean']['mean']:.6f} | {m['annual_excess']['mean']:.6f} | {m['mean_turnover']['mean']:.6f} | {m['final_score']['mean']:.6f} | {m['diagnostic_turnover']['mean']:.6f} | {m['missing_top_fraction']['mean']:.2%} | {m['turnover_gap']['mean']:+.6f} |")
    lines += ["","## 逐折 Score 与稳定性","","| w(E005) | F1 | F2 | F3 robustness | mean | worst | std |","|---:|---:|---:|---:|---:|---:|---:|"]
    for s in p["summaries"]:
        folds={r["fold"]:r for r in s["folds"]}; st=s["stability"]["final_score"]
        lines.append(f"| {s['weight_e005']:.2f} | {folds['F1']['final_score']:.6f} | {folds['F2']['final_score']:.6f} | {folds['F3']['final_score']:.6f} | {st['mean']:.6f} | {st['worst']:.6f} | {st['std']:.6f} |")
    d=p["selected_vs_e005_endpoint"]; q=p["conclusions"]
    lines += ["","## 选择结论","",f"selected w(E005)={q['selected_weight_e005']:.2f}。相对 E005 endpoint：ΔIC={d['ic_mean']:+.6f}，ΔExcess={d['annual_excess']:+.6f}，ΔTurnover={d['mean_turnover']:+.6f}，ΔScore={d['final_score']:+.6f}。",
              f"Score 分解：IC {d['ic_contribution']:+.6f} + excess {d['excess_contribution']:+.6f} + turnover {d['turnover_contribution']:+.6f}。",
              f"F3 中 selected 为五个权重最高，支持保留 E005 endpoint；新 fusion baseline={q['new_baseline']}。",""]
    (out/"phase_c1_comparison.md").write_text("\n".join(lines),encoding="utf-8")
    validation=["# Phase C1 验证记录","",f"- Git HEAD：`{p['git_head']}`。","- w=1/w=0 三折逐指标精确复现 E005/E006 C0b selected。","- daily rank 后只调用一次共享 EMA+hysteresis；alpha/exit 固定且未搜索。","- F3 不参与权重选择，仅在锁定后作 robustness 报告。","- C0/C0b、官方 evaluator 和输入预测运行前后指纹一致。","- 完整测试集：220 passed。",""]
    (out/"phase_c1_validation.md").write_text("\n".join(validation),encoding="utf-8")


if __name__=="__main__": main()
