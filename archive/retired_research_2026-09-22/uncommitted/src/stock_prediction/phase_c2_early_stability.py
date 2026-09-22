"""Early 2020 stability check for the frozen C2 R1/R2 representations."""
from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .baselines import materialize
from .compare_official import load_official
from .folds import split_fold
from .phase_c0 import sha256
from .phase_c1 import complementarity
from .phase_c2 import (evaluate_prediction, make_model, metric_record, postprocess,
                       ranked_feature_indices, rank_matrix_groups_inplace,
                       relevance_and_groups)


def assert_frozen_protocol(config, c2):
    assert config["ranker"] == c2["ranker"]
    assert config["cross_section_input"] == c2["cross_section_input"]
    assert config["postprocess"] == c2["postprocess"] == {"alpha": .3, "exit_fraction": .25}
    assert config["primary_variant"] == "raw"
    assert config["constraints"]["year_2021_unused"] is True


def validation_control(keys, labels, matrix, valid, names):
    return pd.DataFrame({"ts_code": keys["ts_code"][valid].astype(str),
                         "trade_date": keys["trade_date"][valid].astype(int),
                         "y_ret_1d": np.asarray(labels[valid], dtype=float),
                         "flag_limit_up": np.asarray(matrix[valid, names.index("flag_limit_up")])})


def run_candidate(root, candidate, config, feature_config, manifest, evaluator, temp):
    names=manifest["feature_names"]; files=manifest["datasets"]["train"]["files"]
    matrix=np.load(files["matrix"]["path"],mmap_mode="r",allow_pickle=False)
    keys=np.load(files["keys"]["path"],mmap_mode="r",allow_pickle=False)
    labels=np.load(root/"outputs/baselines/cache/labels.npy",mmap_mode="r",allow_pickle=False)
    train,valid,split=split_fold(keys["trade_date"],labels,config["fold"],config["purge_trading_days"])
    assert int(keys["trade_date"][valid].min())>=20200101 and int(keys["trade_date"][valid].max())<=20201231
    assert not np.any(keys["trade_date"][train]>=20210101)
    assert not np.any(keys["trade_date"][valid]>=20210101)
    order,fit_y,groups,_=relevance_and_groups(keys["trade_date"][train],labels[train]); fit=train[order]
    work=Path(temp)/f"{candidate}_train.npy"; x=materialize(matrix,fit,work)
    ranked,preserved=ranked_feature_indices(feature_config,names)
    if candidate=="C2-R2": rank_matrix_groups_inplace(x,groups,ranked)
    model=make_model(config["ranker"]); started=time.perf_counter()
    model.fit(pd.DataFrame(x,columns=names,copy=False),fit_y,group=groups)
    seconds=time.perf_counter()-started; x._mmap.close(); del x; work.unlink()
    pred=np.empty(len(valid),dtype=float); vd=keys["trade_date"][valid]
    if candidate=="C2-R1":
        for start in range(0,len(valid),100_000):
            idx=valid[start:start+100_000]; pred[start:start+len(idx)]=model.predict(pd.DataFrame(matrix[idx],columns=names,copy=False))
    else:
        for date in np.unique(vd):
            pos=np.flatnonzero(vd==date); block=np.asarray(matrix[valid[pos]],dtype=float).copy()
            rank_matrix_groups_inplace(block,[len(block)],ranked)
            pred[pos]=model.predict(pd.DataFrame(block,columns=names,copy=False))
    control=validation_control(keys,labels,matrix,valid,names)
    raw,raw_diag,raw_frame=evaluate_prediction(root,pred,control,evaluator,temp)
    fixed=postprocess(raw_frame,config["postprocess"]["alpha"],config["postprocess"]["exit_fraction"])
    secondary,secondary_diag,_=evaluate_prediction(root,fixed.pred.to_numpy(),control,evaluator,temp)
    records=[metric_record(candidate,"ES2020","raw",raw,raw_diag),metric_record(candidate,"ES2020","postprocessed",secondary,secondary_diag)]
    meta={"candidate":candidate,"split":split,"train_rows":len(train),"valid_rows":len(valid),"group_sum":int(groups.sum()),
          "groups":len(groups),"iterations":model.booster_.current_iteration(),"train_seconds":seconds,
          "ranked_feature_count":len(ranked) if candidate=="C2-R2" else 0,
          "preserved_market_features":[names[i] for i in preserved]}
    return records,meta,raw_frame


def monthly_cache_diagnostics(root):
    out=[]; evaluator=load_official(root)
    with tempfile.TemporaryDirectory(prefix="c2_monthly_") as temp:
      for fold in ["F1","F2","F3"]:
        base=pd.read_csv(root/"outputs/baselines/E005"/fold/"predictions.csv.gz",float_precision="round_trip")
        frames={}
        for candidate in ["C2-R1","C2-R2"]:
            frame=base.copy(); frame["pred"]=np.load(root/"outputs/models/phase_c2"/f"{candidate}_{fold}_pred.npy",allow_pickle=False); frames[candidate]=frame
        for month in sorted((frames["C2-R1"].trade_date//100).unique()):
            a=frames["C2-R1"].loc[frames["C2-R1"].trade_date//100==month].reset_index(drop=True)
            b=frames["C2-R2"].loc[frames["C2-R2"].trade_date//100==month].reset_index(drop=True)
            comp=complementarity(a,b)
            ma=evaluate_prediction(root,a.pred.to_numpy(),a,evaluator,temp)[0]
            mb=evaluate_prediction(root,b.pred.to_numpy(),b,evaluator,temp)[0]
            out.append({"fold":fold,"month":int(month),"delta_ic":mb["ic_mean"]-ma["ic_mean"],
                        "delta_excess":mb["annual_excess"]-ma["annual_excess"],"delta_score":mb["final_score"]-ma["final_score"],
                        "daily_rank_correlation_mean":comp["daily_rank_correlation"]["mean"],
                        "top10_jaccard_mean":comp["daily_top10_overlap"]["jaccard_mean"],
                        "intersection_over_topn_mean":comp["daily_top10_overlap"]["intersection_over_topn_mean"]})
    return out


def write_report(out,p):
    r={f"{x['candidate']}:{x['variant']}":x for x in p["results"]}
    lines=["# C2 early stability：2020 representation check","","Primary comparison uses raw predictions; fixed C0b postprocess is secondary only. 2021 was not used.","",
           "| Candidate | Variant | Rank IC | Annual excess | Turnover | Score | Diagnostic turnover | Missing Top |","|---|---|---:|---:|---:|---:|---:|---:|"]
    for x in p["results"]: lines.append(f"| {x['candidate']} | {x['variant']} | {x['ic_mean']:.6f} | {x['annual_excess']:.6f} | {x['mean_turnover']:.6f} | {x['final_score']:.6f} | {x['diagnostic_turnover']:.6f} | {x['missing_top_fraction']:.2%} |")
    a,b=r["C2-R1:raw"],r["C2-R2:raw"]
    lines += ["","## Raw R2 minus R1","",f"- ΔIC: {b['ic_mean']-a['ic_mean']:+.6f}",f"- Δannual excess: {b['annual_excess']-a['annual_excess']:+.6f}",f"- Δturnover: {b['mean_turnover']-a['mean_turnover']:+.6f}",f"- ΔScore: {b['final_score']-a['final_score']:+.6f}",""]
    (out/"phase_c2_early_stability_comparison.md").write_text("\n".join(lines),encoding="utf-8")


def main():
    root=Path.cwd(); out=root/"outputs/phase_c2_early_stability"; out.mkdir(parents=True,exist_ok=True)
    path=root/"config/phase_c2_early_stability.yaml"; config=yaml.safe_load(path.read_text(encoding="utf-8"))
    c2=yaml.safe_load((root/"config/phase_c2_lambdarank.yaml").read_text(encoding="utf-8")); assert_frozen_protocol(config,c2)
    feature_config=yaml.safe_load((root/"config/features_v1.yaml").read_text(encoding="utf-8")); manifest=json.loads((root/"outputs/full147_manifest.json").read_text(encoding="utf-8"))
    protected=[root/"evaluate.py",root/"reference/evaluate_official.py",root/"outputs/phase_c2/phase_c2_comparison.json"]
    before={str(p.relative_to(root)):sha256(p) for p in protected}; evaluator=load_official(root); results=[]; metadata=[]
    with tempfile.TemporaryDirectory(prefix="c2_early_") as temp:
        for candidate in config["candidates"]:
            rows,meta,_=run_candidate(root,candidate,config,feature_config,manifest,evaluator,temp); results.extend(rows); metadata.append(meta); print(f"DONE {candidate}/ES2020",flush=True)
    monthly=monthly_cache_diagnostics(root)
    after={str(p.relative_to(root)):sha256(p) for p in protected}; assert before==after
    payload={"phase":config["phase"],"git_head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
             "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"config_sha256":sha256(path),"config":config,
             "results":results,"training_metadata":metadata,"existing_c2_monthly_diagnostics":monthly,
             "verification":{"year_2021_unused":True,"existing_folds_retrained":False,"protected_artifacts_unchanged":True}}
    (out/"phase_c2_early_stability_comparison.json").write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    write_report(out,payload)


if __name__=="__main__": main()
