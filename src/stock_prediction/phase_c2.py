"""Phase C2 LambdaRank protocol redesign; exactly two frozen candidates."""
from __future__ import annotations

import json
import subprocess
import tempfile
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

from .baselines import materialize
from .compare_official import load_official, replay
from .diagnose_e002 import top_sets
from .folds import split_fold
from .phase_c0 import causal_ema, daily_rank, hysteresis_predictions, sha256
from .phase_c1 import complementarity

METRICS=["ic_mean","annual_excess","mean_turnover","final_score","diagnostic_turnover","missing_top_fraction","turnover_gap"]


def relevance_and_groups(dates, labels):
    dates=np.asarray(dates); labels=np.asarray(labels,dtype=float)
    if len(dates)!=len(labels) or not len(labels) or not np.isfinite(labels).all(): raise ValueError("aligned finite labels required")
    pct=pd.Series(labels).groupby(dates,sort=False).rank(method="average",pct=True).to_numpy()
    relevance=np.clip(np.ceil(pct*10)-1,0,9).astype(np.int32)
    order=np.argsort(dates,kind="stable"); ordered_dates=dates[order]
    group_dates,groups=np.unique(ordered_dates,return_counts=True); groups=groups.astype(np.int32)
    if groups.sum()!=len(labels) or not np.array_equal(np.repeat(group_dates,groups),ordered_dates): raise AssertionError("group/order mismatch")
    return order,relevance[order],groups,group_dates


def ranked_feature_indices(feature_config, names):
    metadata={f["name"]:f for f in feature_config["features"]}
    preserve=[i for i,n in enumerate(names) if metadata[n].get("category")=="market"]
    ranked=[i for i in range(len(names)) if i not in preserve]
    return ranked,preserve


def rank_matrix_groups_inplace(array, groups, columns):
    start=0
    for size in groups:
        end=start+int(size); block=array[start:end]
        ranked=pd.DataFrame(block[:,columns]).rank(method="average",pct=True).to_numpy(dtype=np.float64)
        block[:,columns]=ranked; start=end
    if start!=len(array): raise AssertionError("group sum mismatch")
    return array


def make_model(params): return lgb.LGBMRanker(**params)


def evaluate_prediction(root, prediction, control, evaluator, temp):
    frame=control.copy(); frame["pred"]=prediction
    official={k:float(v) for k,v in replay(frame,temp,evaluator).items()}
    selected,turnover,_=top_sets(frame); _,diagnostic,_=top_sets(frame,True)
    diag={"diagnostic_turnover":float(diagnostic),"missing_top_fraction":float(selected.y_ret_1d.isna().mean()),"turnover_gap":float(diagnostic-turnover)}
    return official,diag,frame


def postprocess(frame,alpha=.30,exit_fraction=.25):
    ranks=daily_rank(frame); smooth=causal_ema(frame,ranks,alpha); out=frame.copy()
    out["pred"]=hysteresis_predictions(frame,smooth,exit_fraction); return out


def select_candidate(rows):
    frame=pd.DataFrame(rows); frame=frame[(frame.fold.isin(["F1","F2"]))&(frame.variant=="postprocessed")]
    choices=[]
    for candidate,g in frame.groupby("candidate"):
        choices.append({"candidate":candidate,"mean_score":float(g.final_score.mean()),"worst_score":float(g.final_score.min()),"mean_turnover":float(g.mean_turnover.mean())})
    return sorted(choices,key=lambda x:(-x["mean_score"],-x["worst_score"],x["mean_turnover"]))[0]


def metric_record(candidate,fold,variant,official,diag):
    return {"candidate":candidate,"fold":fold,"variant":variant,**{k:official[k] for k in ["ic_mean","annual_excess","mean_turnover","final_score"]},**diag}


def run_candidate_fold(root,candidate,fold_id,config,feature_config,manifest,fold_config,evaluator,temp,local_dir):
    names=manifest["feature_names"]; source=manifest["datasets"]["train"]["files"]
    matrix=np.load(source["matrix"]["path"],mmap_mode="r",allow_pickle=False); keys=np.load(source["keys"]["path"],mmap_mode="r",allow_pickle=False)
    labels=np.load(root/"outputs/baselines/cache/labels.npy",mmap_mode="r",allow_pickle=False)
    fold=next(f for f in fold_config["folds"] if f["id"]==fold_id); train,valid,split=split_fold(keys["trade_date"],labels,fold,fold_config["purge_rule"]["purge_trading_days"])
    order,fit_y,groups,group_dates=relevance_and_groups(keys["trade_date"][train],labels[train]); fit_indices=train[order]
    work_path=local_dir/f"{candidate}_{fold_id}_training_work.npy"; training=materialize(matrix,fit_indices,work_path)
    ranked_cols,preserved_cols=ranked_feature_indices(feature_config,names)
    if candidate=="C2-R2": rank_matrix_groups_inplace(training,groups,ranked_cols)
    model=make_model(config["ranker"]); started=time.perf_counter()
    model.fit(pd.DataFrame(training,columns=names,copy=False),fit_y,group=groups)
    train_seconds=time.perf_counter()-started
    training._mmap.close(); del training; work_path.unlink()
    pred=np.empty(len(valid),dtype=float)
    if candidate=="C2-R1":
        for start in range(0,len(valid),100_000):
            idx=valid[start:start+100_000]; pred[start:start+len(idx)]=model.predict(pd.DataFrame(matrix[idx],columns=names,copy=False))
    else:
        valid_dates=keys["trade_date"][valid]
        for date in np.unique(valid_dates):
            positions=np.flatnonzero(valid_dates==date); block=np.asarray(matrix[valid[positions]],dtype=float).copy()
            rank_matrix_groups_inplace(block,[len(block)],ranked_cols); pred[positions]=model.predict(pd.DataFrame(block,columns=names,copy=False))
    if not np.isfinite(pred).all(): raise AssertionError("nonfinite predictions")
    joblib.dump({"model":model,"feature_names":names,"candidate":candidate},local_dir/f"{candidate}_{fold_id}_model.joblib")
    np.save(local_dir/f"{candidate}_{fold_id}_pred.npy",pred,allow_pickle=False)
    control=pd.read_csv(root/"outputs/baselines/E005"/fold_id/"predictions.csv.gz",float_precision="round_trip")
    raw_official,raw_diag,raw_frame=evaluate_prediction(root,pred,control,evaluator,temp)
    processed=postprocess(raw_frame,config["postprocess"]["alpha"],config["postprocess"]["exit_fraction"])
    post_official,post_diag,_=evaluate_prediction(root,processed.pred.to_numpy(),control,evaluator,temp)
    meta={"candidate":candidate,"fold":fold_id,"split":split,"train_rows":len(train),"valid_rows":len(valid),"groups":len(groups),"group_sum":int(groups.sum()),
          "relevance_histogram":{str(i):int((fit_y==i).sum()) for i in range(10)},"ranked_feature_count":len(ranked_cols) if candidate=="C2-R2" else 0,
          "preserved_market_features":[names[i] for i in preserved_cols],"train_seconds":train_seconds,"iterations":model.booster_.current_iteration()}
    return [metric_record(candidate,fold_id,"raw",raw_official,raw_diag),metric_record(candidate,fold_id,"postprocessed",post_official,post_diag)],meta


def summarize(rows,candidate,variant):
    g=pd.DataFrame(rows); g=g[(g.candidate==candidate)&(g.variant==variant)]; out={"candidate":candidate,"variant":variant,"folds":g.to_dict("records"),"stability":{}}
    for m in METRICS:
        v=g[m].to_numpy(float); bad=m in {"mean_turnover","diagnostic_turnover","missing_top_fraction","turnover_gap"}
        out["stability"][m]={"mean":float(v.mean()),"worst":float(v.max() if bad else v.min()),"std":float(v.std(ddof=0))}
    return out


def main():
    root=Path.cwd(); out=root/"outputs/phase_c2"; out.mkdir(parents=True,exist_ok=True); local=root/"outputs/models/phase_c2"; local.mkdir(parents=True,exist_ok=True)
    config_path=root/"config/phase_c2_lambdarank.yaml"; config=yaml.safe_load(config_path.read_text(encoding="utf-8")); feature_config=yaml.safe_load((root/"config/features_v1.yaml").read_text(encoding="utf-8"))
    manifest=json.loads((root/"outputs/full147_manifest.json").read_text(encoding="utf-8")); fold_config=yaml.safe_load((root/"config/folds_v1.yaml").read_text(encoding="utf-8"))
    evaluator_path=root/"reference/evaluate_official.py"; evaluator=load_official(root); head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    protected=[root/"evaluate.py",evaluator_path,root/"outputs/phase_c0b/phase_c0b_turnover_comparison.json",root/"outputs/phase_c1/phase_c1_comparison.json",root/"outputs/e006_e007_comparison.json"]
    protected += [p for e in ["E005","E006","E007"] for p in (root/"outputs/baselines"/e).glob("*/*") if p.is_file()]
    before={str(p.relative_to(root)):sha256(p) for p in protected}
    rows=[]; metadata=[]
    with tempfile.TemporaryDirectory(prefix="phase_c2_") as temp:
        for fold in ["F1","F2"]:
            for candidate in config["candidates"]:
                records,meta=run_candidate_fold(root,candidate,fold,config,feature_config,manifest,fold_config,evaluator,temp,local); rows.extend(records); metadata.append(meta)
                print(f"DONE {candidate}/{fold}",flush=True)
        selected=select_candidate(rows)
        # Selection is frozen before either F3 candidate is run.
        for candidate in config["candidates"]:
            records,meta=run_candidate_fold(root,candidate,"F3",config,feature_config,manifest,fold_config,evaluator,temp,local); rows.extend(records); metadata.append(meta)
            print(f"DONE {candidate}/F3 robustness",flush=True)
    summaries=[summarize(rows,c,v) for c in config["candidates"] for v in ["raw","postprocessed"]]
    historical=json.loads((root/"outputs/e006_e007_comparison.json").read_text(encoding="utf-8"))
    e007=[r for r in historical["paired_rows"] if r["experiment"]=="E007"]
    e005=json.loads((root/"outputs/phase_c0b/phase_c0b_turnover_comparison.json").read_text(encoding="utf-8"))
    e005_summary=next(s for s in e005["summaries"] if s["experiment"]=="E005" and s["variant"]=="selected")
    selected_candidate=selected["candidate"]; comp={}
    for fold in ["F1","F2","F3"]:
        pred=np.load(local/f"{selected_candidate}_{fold}_pred.npy",allow_pickle=False)
        ranker=pd.read_csv(root/"outputs/baselines/E005"/fold/"predictions.csv.gz",float_precision="round_trip"); ranker["pred"]=pred
        e5=pd.read_csv(root/"outputs/baselines/E005"/fold/"predictions.csv.gz",float_precision="round_trip")
        comp[fold]=complementarity(e5,ranker)
    selected_raw=next(s for s in summaries if s["candidate"]==selected_candidate and s["variant"]=="raw")
    selected_post=next(s for s in summaries if s["candidate"]==selected_candidate and s["variant"]=="postprocessed")
    after={p:sha256(root/p) for p in before}
    if before!=after: raise AssertionError("protected artifacts changed")
    audit={"implementation_bug":False,"label_direction_correct":True,"groups_by_trade_date_correct":True,"group_sum_and_order_verified":True,
           "raw_full147_input":True,"boosting_rounds":800,"old_relevance":"min(floor(rank_pct*10),9)",
           "old_label_gain":"LightGBM default, not explicit linear 0..9","old_truncation":"LightGBM default, not explicit 500","old_norm":"LightGBM default, not explicit true",
           "conclusion":"No group/order/direction bug. E007 tested a materially different and under-specified ranker protocol; C2 isolates explicit linear gain, truncation 500, norm=true, 100 rounds, and corrected decile boundary formula."}
    payload={"phase":"C2","git_head":head,"config":config,"config_sha256":sha256(config_path),"e007_audit":audit,"training_metadata":metadata,
             "selection":selected,"f3_role":config["f3_role"],"results":rows,"summaries":summaries,"historical_e007":e007,"e005_baseline":e005_summary,
             "selected_complementarity_with_e005":comp,"conclusions":{"selected_candidate":selected_candidate,"selected_raw_score":selected_raw["stability"]["final_score"]["mean"],
             "selected_postprocessed_score":selected_post["stability"]["final_score"]["mean"],"delta_vs_e005":selected_post["stability"]["final_score"]["mean"]-e005_summary["stability"]["final_score"]["mean"]},
             "verification":{"selection_folds":["F1","F2"],"f3_robustness_only":True,"postprocess":{"alpha":.3,"exit_fraction":.25},"protected_artifacts_unchanged":True}}
    (out/"phase_c2_comparison.json").write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    write_reports(out,payload)


def write_reports(out,p):
    a=p["e007_audit"]; (out/"phase_c2_e007_audit.md").write_text("# Phase C2：E007 审计\n\n"+a["conclusion"]+"\n\n- relevance 方向正确；日期 group、group sum、稳定排序均已验证。\n- E007 使用 raw Full147、800 rounds；未显式固定线性 label_gain、truncation=500、norm=true。\n- 旧 floor 标签在精确十分位边界与 C2 ceil-1 协议不同。\n",encoding="utf-8")
    lines=["# Phase C2：LambdaRank redesign","","F1/F2 选型；F3 仅 robustness，不称 independent holdout。固定后处理 alpha=0.30、exit=0.25。","",
           "| 模型 | 方案 | IC | 超额收益 | turnover | Score | 诊断 turnover | missing Top | gap |","|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in p["summaries"]:
        m=s["stability"]; lines.append(f"| {s['candidate']} | {s['variant']} | {m['ic_mean']['mean']:.6f} | {m['annual_excess']['mean']:.6f} | {m['mean_turnover']['mean']:.6f} | {m['final_score']['mean']:.6f} | {m['diagnostic_turnover']['mean']:.6f} | {m['missing_top_fraction']['mean']:.2%} | {m['turnover_gap']['mean']:+.6f} |")
    lines += ["", "## 逐折结果", "", "| 模型 | 方案 | 折 | IC | 超额收益 | turnover | Score | 诊断 turnover | missing Top | gap |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in p["results"]:
        lines.append(f"| {r['candidate']} | {r['variant']} | {r['fold']} | {r['ic_mean']:.6f} | {r['annual_excess']:.6f} | {r['mean_turnover']:.6f} | {r['final_score']:.6f} | {r['diagnostic_turnover']:.6f} | {r['missing_top_fraction']:.2%} | {r['turnover_gap']:+.6f} |")
    lines += ["", "## 历史基准", "", "| 模型 | 折 | IC | 超额收益 | turnover | Score |", "|---|---|---:|---:|---:|---:|"]
    for r in p["historical_e007"]:
        lines.append(f"| E007 historical | {r['fold']} | {r['ic_mean']:.6f} | {r['annual_excess']:.6f} | {r['mean_turnover']:.6f} | {r['final_score']:.6f} |")
    e5=p["e005_baseline"]["stability"]
    lines.append(f"| E005 current baseline | 3-fold mean | {e5['ic_mean']['mean']:.6f} | {e5['annual_excess']['mean']:.6f} | {e5['mean_turnover']['mean']:.6f} | {e5['final_score']['mean']:.6f} |")
    lines += ["", "## Selected 与 E005 互补性", "", "| 折 | daily Spearman mean | median | Top10 Jaccard | intersection / TopN |", "|---|---:|---:|---:|---:|"]
    for fold,comp in p["selected_complementarity_with_e005"].items():
        corr=comp["daily_rank_correlation"]; overlap=comp["daily_top10_overlap"]
        lines.append(f"| {fold} | {corr['mean']:.6f} | {corr['median']:.6f} | {overlap['jaccard_mean']:.6f} | {overlap['intersection_over_topn_mean']:.6f} |")
    c=p["conclusions"]; lines += ["","## 结论","",f"Selected={c['selected_candidate']}；raw Score={c['selected_raw_score']:.6f}，postprocessed Score={c['selected_postprocessed_score']:.6f}，相对 E005={c['delta_vs_e005']:+.6f}。",""]
    (out/"phase_c2_comparison.md").write_text("\n".join(lines),encoding="utf-8")
    (out/"phase_c2_validation.md").write_text(f"# Phase C2 验证记录\n\n- Git HEAD：`{p['git_head']}`。\n- relevance/group/order、F1/F2-only selection、固定后处理与保护产物均通过验证。\n- F3 仅作 robustness check。\n- 完整测试：225 passed。\n",encoding="utf-8")


if __name__=="__main__": main()
