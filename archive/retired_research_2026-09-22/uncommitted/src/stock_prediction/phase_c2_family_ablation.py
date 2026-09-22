"""Round-one metadata-defined feature-family ablation for frozen C2-R1."""
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

from .compare_official import load_official
from .folds import split_fold
from .phase_c0 import sha256
from .phase_c2 import evaluate_prediction, make_model, relevance_and_groups
from .phase_c2_early_stability import validation_control


ALLOWED_CATEGORIES={"cross_section_rank","relative","market"}


def feature_set_indices(feature_config,names,remove_category):
    metadata={f["name"]:f for f in feature_config["features"]}
    if set(names)-set(metadata): raise ValueError("feature metadata incomplete")
    if remove_category is not None and remove_category not in ALLOWED_CATEGORIES: raise ValueError("category outside frozen ablation scope")
    removed=[i for i,n in enumerate(names) if metadata[n]["category"]==remove_category] if remove_category else []
    kept=[i for i in range(len(names)) if i not in removed]
    return kept,removed


def materialize_subset(matrix,rows,columns,path,batch_size=100_000):
    target=np.lib.format.open_memmap(path,mode="w+",dtype="float64",shape=(len(rows),len(columns)))
    for start in range(0,len(rows),batch_size):
        idx=rows[start:start+batch_size]; target[start:start+len(idx)]=matrix[np.ix_(idx,columns)]
    target.flush(); return target


def classify_family(deltas):
    scores=[x["delta_score"] for x in deltas]
    if all(x<0 for x in scores): return "retain"
    if all(x>0 for x in scores): return "deletion_candidate"
    return "unstable_retain_for_now"


def assert_scope(config,c2):
    assert config["primary_variant"]=="raw" and config["purge_trading_days"]==1
    assert [x["remove_category"] for x in config["feature_sets"]]==[None,"cross_section_rank","relative","market"]
    assert config["constraints"]["year_2021_unused"] is True
    return c2["ranker"]


def run_one(root,spec,fold,ranker,feature_config,manifest,evaluator,temp):
    names=manifest["feature_names"]; files=manifest["datasets"]["train"]["files"]
    matrix=np.load(files["matrix"]["path"],mmap_mode="r",allow_pickle=False); keys=np.load(files["keys"]["path"],mmap_mode="r",allow_pickle=False)
    labels=np.load(root/"outputs/baselines/cache/labels.npy",mmap_mode="r",allow_pickle=False)
    train,valid,split=split_fold(keys["trade_date"],labels,fold,1)
    assert keys["trade_date"][train].max()<20210101 and keys["trade_date"][valid].max()<20210101
    order,fit_y,groups,_=relevance_and_groups(keys["trade_date"][train],labels[train]); fit=train[order]
    kept,removed=feature_set_indices(feature_config,names,spec["remove_category"])
    path=Path(temp)/f"{spec['id']}_{fold['id']}.npy"; x=materialize_subset(matrix,fit,kept,path)
    model=make_model(ranker); started=time.perf_counter(); selected_names=[names[i] for i in kept]
    model.fit(pd.DataFrame(x,columns=selected_names,copy=False),fit_y,group=groups); seconds=time.perf_counter()-started
    x._mmap.close(); del x; path.unlink(); pred=np.empty(len(valid),float)
    for start in range(0,len(valid),100_000):
        idx=valid[start:start+100_000]; block=matrix[np.ix_(idx,kept)]
        pred[start:start+len(idx)]=model.predict(pd.DataFrame(block,columns=selected_names,copy=False))
    control=validation_control(keys,labels,matrix,valid,names); official,diag,_=evaluate_prediction(root,pred,control,evaluator,temp)
    row={"feature_set":spec["id"],"removed_category":spec["remove_category"],"fold":fold["id"],
         **{k:official[k] for k in ["ic_mean","annual_excess","mean_turnover","final_score"]},**diag}
    meta={"feature_set":spec["id"],"fold":fold["id"],"split":split,"feature_count":len(kept),"removed_feature_count":len(removed),
          "removed_features":[names[i] for i in removed],"group_sum":int(groups.sum()),"groups":len(groups),
          "iterations":model.booster_.current_iteration(),"train_seconds":seconds}
    return row,meta


def add_deltas_and_decisions(rows):
    baseline={x["fold"]:x for x in rows if x["feature_set"]=="Full147"}; deltas=[]; decisions={}
    for row in rows:
        if row["feature_set"]=="Full147": continue
        b=baseline[row["fold"]]
        delta={"feature_set":row["feature_set"],"removed_category":row["removed_category"],"fold":row["fold"],
               "delta_ic":row["ic_mean"]-b["ic_mean"],"delta_excess":row["annual_excess"]-b["annual_excess"],
               "delta_turnover":row["mean_turnover"]-b["mean_turnover"],"delta_score":row["final_score"]-b["final_score"]}
        deltas.append(delta)
    for name in sorted({x["feature_set"] for x in deltas}):
        subset=[x for x in deltas if x["feature_set"]==name]; decisions[name]=classify_family(subset)
    return deltas,decisions


def write_report(out,p):
    d={(x["feature_set"],x["fold"]):x for x in p["deltas"]}
    lines=["# C2 R1 feature-family ablation — round 1","","Raw prediction only. 2019 and 2020 are the only OOT validation years; 2021 is unused.","",
           "| Feature set | Fold | IC | Excess | Turnover | Score | Diagnostic turnover | Missing Top | ΔIC | Δexcess | Δturnover | ΔScore |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in p["results"]:
        x=d.get((r["feature_set"],r["fold"])); ds=" | ".join(f"{x[k]:+.6f}" for k in ["delta_ic","delta_excess","delta_turnover","delta_score"]) if x else "— | — | — | —"
        lines.append(f"| {r['feature_set']} | {r['fold']} | {r['ic_mean']:.6f} | {r['annual_excess']:.6f} | {r['mean_turnover']:.6f} | {r['final_score']:.6f} | {r['diagnostic_turnover']:.6f} | {r['missing_top_fraction']:.2%} | {ds} |")
    lines += ["","## Frozen-rule decisions","","| Ablation | Decision |","|---|---|"]
    for name,value in p["decisions"].items(): lines.append(f"| {name} | {value} |")
    (out/"phase_c2_family_ablation_comparison.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    root=Path.cwd(); out=root/"outputs/phase_c2_family_ablation"; out.mkdir(parents=True,exist_ok=True)
    path=root/"config/phase_c2_family_ablation.yaml"; config=yaml.safe_load(path.read_text(encoding="utf-8")); c2=yaml.safe_load((root/"config/phase_c2_lambdarank.yaml").read_text(encoding="utf-8")); ranker=assert_scope(config,c2)
    features=yaml.safe_load((root/"config/features_v1.yaml").read_text(encoding="utf-8")); manifest=json.loads((root/"outputs/full147_manifest.json").read_text(encoding="utf-8"))
    protected=[root/"evaluate.py",root/"reference/evaluate_official.py",root/"outputs/phase_c2/phase_c2_comparison.json",root/"outputs/phase_c2_early_stability/phase_c2_early_stability_comparison.json"]
    before={str(p.relative_to(root)):sha256(p) for p in protected}; evaluator=load_official(root); rows=[]; metadata=[]
    with tempfile.TemporaryDirectory(prefix="c2_ablation_") as temp:
        for fold in config["folds"]:
            for spec in config["feature_sets"]:
                row,meta=run_one(root,spec,fold,ranker,features,manifest,evaluator,temp); rows.append(row); metadata.append(meta); print(f"DONE {spec['id']}/{fold['id']}",flush=True)
    deltas,decisions=add_deltas_and_decisions(rows); after={str(p.relative_to(root)):sha256(p) for p in protected}; assert before==after
    payload={"phase":config["phase"],"git_head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "config_sha256":sha256(path),"config":config,"ranker":ranker,"results":rows,"deltas":deltas,"decisions":decisions,"training_metadata":metadata,
             "verification":{"year_2021_unused":True,"primary_raw_only":True,"protected_artifacts_unchanged":True}}
    (out/"phase_c2_family_ablation_comparison.json").write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8"); write_report(out,payload)


if __name__=="__main__": main()
