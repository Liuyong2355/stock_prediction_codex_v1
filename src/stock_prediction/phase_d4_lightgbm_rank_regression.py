"""D4: replace only the D0 XGBoost regressor with frozen E004 LightGBM."""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

from .baselines import configs, peak_memory_gb, save_json, save_lightgbm_text
from .build_basic40 import sha256
from .compare_official import load_official, replay
from .e003 import official_daily_and_diagnostic
from .folds import split_fold
from .phase_d0_e006_rank_view import materialize_augmented, resolve_rank_sources
from .phase_d2b_top10_binary import complementarity
from .ranking import iterations, predict_batches
from .swanlab_tracker import start_swanlab
from .targets import training_target


OFFICIAL_METRICS = ("ic_mean", "annual_excess", "mean_turnover", "final_score")
REPORT_METRICS = OFFICIAL_METRICS + ("diagnostic_turnover", "missing_top_fraction")


def make_lightgbm_model(experiments):
    entry = next(item for item in experiments["experiments"] if item["id"] == "E004")
    if (entry["feature_set"], entry["model"], entry["target"]) != ("Full147", "lightgbm_reg_v1", "rank"):
        raise AssertionError("Frozen E004 definition changed")
    params = dict(experiments["model_catalog"]["lightgbm_reg_v1"]["params"])
    return lgb.LGBMRegressor(**params), params


def d4_complementarity(candidate, baseline):
    metrics, daily = complementarity(candidate, baseline)
    daily["top10_exclusive_union_fraction"] = 1.0 - daily["top10_jaccard"]
    daily["top10_per_model_exclusive_fraction"] = 1.0 - daily["top10_overlap"]
    for metric in ("top10_exclusive_union_fraction", "top10_per_model_exclusive_fraction"):
        metrics[metric] = {"mean": float(daily[metric].mean()), "worst": float(daily[metric].max()),
                           "std_ddof0": float(daily[metric].std(ddof=0))}
    return metrics, daily


def evaluate_frame(root, frame):
    with tempfile.TemporaryDirectory(prefix="phase_d4_official_") as temp:
        official = replay(frame, temp, load_official(root))
    _, diagnostic, _, _ = official_daily_and_diagnostic(frame)
    return official, diagnostic


def run_fold(root: Path, fold_id: str, stage: dict) -> dict:
    fold_config, experiments = configs(root)
    feature_stage = yaml.safe_load((root / stage["feature_config"]).read_text(encoding="utf-8"))
    manifest = json.loads((root / "outputs/full147_manifest.json").read_text(encoding="utf-8"))
    names = list(manifest["feature_names"])
    new_sources, new_names = resolve_rank_sources(feature_stage, names)
    source_columns = [names.index(name) for name in new_sources]
    matrix = np.load(manifest["datasets"]["train"]["files"]["matrix"]["path"], mmap_mode="r")
    keys = np.load(manifest["datasets"]["train"]["files"]["keys"]["path"], mmap_mode="r")
    labels = np.load(root / "outputs/baselines/cache/labels.npy", mmap_mode="r")
    fold = next(item for item in fold_config["folds"] if item["id"] == fold_id)
    train, valid, split = split_fold(keys["trade_date"], labels, fold, fold_config["purge_rule"]["purge_trading_days"])
    baseline = json.loads((root / "outputs/phase_d0_e006_rank_view" / fold_id / "result.json").read_text(encoding="utf-8"))
    if split != baseline["split"] or len(train) != baseline["train_rows"] or len(valid) != baseline["valid_rows"]:
        raise AssertionError("D0 train/validation split changed")
    fit_y = training_target(keys["trade_date"][train], labels[train], "rank")
    model, model_params = make_lightgbm_model(experiments)
    tracker = start_swanlab(
        root, f"D4_{fold_id}", stage["swanlab"]["group"],
        {"phase": "D4", "fold": fold_id, "single_variable": "model_family",
         "feature_count": len(names) + len(new_names), "feature_view": stage["feature_view"],
         "model": stage["model"], "model_params": model_params, "target": "rank",
         "seed": experiments["seed"], "split": split, "train_rows": len(train), "valid_rows": len(valid),
         "postprocess": None, "official_evaluator_sha256": sha256(root / "evaluate.py")},
        description="D4 frozen LightGBM rank regression; aggregate metrics only",
    )
    out = root / "outputs/phase_d4_lightgbm_rank_regression" / fold_id
    out.mkdir(parents=True, exist_ok=True)
    if (out / "result.json").exists():
        tracker.finish(state="aborted", error="completed fold already exists")
        raise FileExistsError(f"Completed fold exists: {fold_id}")
    started = time.perf_counter()
    train_path, valid_path = out / "training_work.npy", out / "validation_work.npy"
    try:
        training = materialize_augmented(matrix, keys, train, source_columns, train_path)
        validation = materialize_augmented(matrix, keys, valid, source_columns, valid_path)
        all_names = names + new_names
        np.random.seed(experiments["seed"])
        fit_started = time.perf_counter()
        model.fit(pd.DataFrame(training, columns=all_names, copy=False), fit_y)
        fit_seconds = time.perf_counter() - fit_started
        if iterations(model) != 800:
            raise AssertionError("Frozen E004 iteration count changed")
        pred = predict_batches(model, validation, np.arange(len(valid)), all_names)
        if not np.isfinite(pred).all():
            raise AssertionError("Non-finite D4 predictions")
        frame = pd.DataFrame({"ts_code": keys["ts_code"][valid], "trade_date": keys["trade_date"][valid],
                              "pred": pred, "y_ret_1d": np.asarray(labels[valid]),
                              "flag_limit_up": matrix[valid, names.index("flag_limit_up")]})
        frame.to_csv(out / "predictions.csv.gz", index=False, float_format="%.17g",
                     compression={"method": "gzip", "mtime": 0})
        official, diagnostic = evaluate_frame(root, frame)
        d0_predictions = pd.read_csv(root / "outputs/phase_d0_e006_rank_view" / fold_id / "predictions.csv.gz",
                                     float_precision="round_trip")
        comp, daily_comp = d4_complementarity(frame, d0_predictions)
        daily_comp.to_csv(out / "daily_complementarity.csv", index=False, float_format="%.17g")
        save_lightgbm_text(model, out / "model.txt")
        for item in (training, validation):
            item._mmap.close()
        train_path.unlink(); valid_path.unlink()
        result = {
            "phase": "D4", "fold": fold_id, "baseline": "D0 Candidate",
            "single_variable": "model_family", "feature_view": stage["feature_view"],
            "feature_count": len(all_names), "model": "lightgbm_reg_v1", "model_params": model_params,
            "target": "rank", "split": split, "train_rows": len(train), "valid_rows": len(valid),
            "official": official, "diagnostic": diagnostic, "complementarity": comp,
            "baseline_official": baseline["official"], "baseline_diagnostic": baseline["diagnostic"],
            "delta_candidate_minus_d0": {key: float(official[key] - baseline["official"][key]) for key in OFFICIAL_METRICS},
            "fit_seconds": fit_seconds, "elapsed_seconds": time.perf_counter() - started,
            "peak_memory_gb": peak_memory_gb(), "seed": experiments["seed"], "postprocess": None,
            "config_sha256": sha256(root / "config/phase_d4_lightgbm_rank_regression.yaml"),
            "feature_config_sha256": sha256(root / stage["feature_config"]),
            "official_evaluator_sha256": sha256(root / "evaluate.py"),
        }
        save_json(out / "result.json", result)
        tracker.log({"official": {k: official[k] for k in OFFICIAL_METRICS},
                     "diagnostic": {"turnover": diagnostic["diagnostic_exclude_missing_y_turnover"],
                                    "missing_top_fraction": diagnostic["missing_share_of_top"]},
                     "delta_vs_d0": result["delta_candidate_minus_d0"],
                     "complementarity": {k: v["mean"] for k, v in comp.items()},
                     "runtime": {"fit_seconds": fit_seconds, "elapsed_seconds": result["elapsed_seconds"]}})
        tracker.finish()
        return result
    except Exception as exc:
        tracker.finish(state="crashed", error=str(exc))
        raise


def summarize(root: Path, stage: dict) -> dict:
    rows = [json.loads((root / "outputs/phase_d4_lightgbm_rank_regression" / f / "result.json").read_text(encoding="utf-8")) for f in stage["folds"]]
    def value(row, metric, baseline=False):
        if metric == "diagnostic_turnover":
            return row["baseline_diagnostic" if baseline else "diagnostic"]["diagnostic_exclude_missing_y_turnover"]
        if metric == "missing_top_fraction":
            return row["baseline_diagnostic" if baseline else "diagnostic"]["missing_share_of_top"]
        return row["baseline_official" if baseline else "official"][metric]
    summary = {}
    for metric in REPORT_METRICS:
        candidate = np.array([value(row, metric) for row in rows], dtype=float)
        baseline = np.array([value(row, metric, True) for row in rows], dtype=float)
        lower_better = "turnover" in metric or metric == "missing_top_fraction"
        summary[metric] = {
            "d0_mean": float(baseline.mean()), "d0_worst": float(baseline.max() if lower_better else baseline.min()),
            "d0_std_ddof0": float(baseline.std(ddof=0)), "candidate_mean": float(candidate.mean()),
            "candidate_worst": float(candidate.max() if lower_better else candidate.min()),
            "candidate_std_ddof0": float(candidate.std(ddof=0)), "mean_delta": float((candidate-baseline).mean()),
            "improved_folds": int(((candidate < baseline) if lower_better else (candidate > baseline)).sum()),
            "folds": {row["fold"]: {"d0": float(b), "candidate": float(c), "delta": float(c-b)}
                      for row, b, c in zip(rows, baseline, candidate)},
        }
    comp_metrics = ("prediction_spearman", "top10_jaccard", "top10_overlap",
                    "top10_exclusive_union_fraction", "top10_per_model_exclusive_fraction")
    comp_summary = {}
    for metric in comp_metrics:
        fold_means = np.array([row["complementarity"][metric]["mean"] for row in rows])
        comp_summary[metric] = {"mean_of_fold_means": float(fold_means.mean()),
                                "highest_fold": float(fold_means.max()),
                                "std_across_folds_ddof0": float(fold_means.std(ddof=0)),
                                "folds": {row["fold"]: float(v) for row, v in zip(rows, fold_means)}}
    stronger_main = (summary["final_score"]["candidate_mean"] > 0.240126
                     and summary["final_score"]["improved_folds"] >= 2
                     and summary["ic_mean"]["mean_delta"] > 0 and summary["annual_excess"]["mean_delta"] > 0)
    second_model = (min(summary["ic_mean"]["folds"][f]["candidate"] for f in stage["folds"]) > 0
                    and min(summary["annual_excess"]["folds"][f]["candidate"] for f in stage["folds"]) > 0
                    and min(summary["final_score"]["folds"][f]["candidate"] for f in stage["folds"]) > 0
                    and summary["missing_top_fraction"]["candidate_mean"] < 0.01)
    payload = {"phase": "D4", "comparison": "D0 XGBoost vs frozen LightGBM rank regression",
               "fold_results": rows, "summary": summary, "complementarity": comp_summary,
               "stronger_main_model": stronger_main, "second_model_candidate": second_model,
               "decision": "promote_main_candidate" if stronger_main else ("retain_second_model_no_fusion" if second_model else "stop_plain_lightgbm_regression")}
    out = root / "outputs/phase_d4_lightgbm_rank_regression"
    save_json(out / "comparison.json", payload)
    lines = ["# Phase D4：LightGBM Rank Regression", "",
             "唯一变量：D0 XGBoost Regressor 替换为冻结 E004 LightGBM Regressor；无调参、postprocess 或融合。", "",
             "| Metric | D0 mean | D0 worst | D0 std | D4 mean | D4 worst | D4 std | Mean Δ | Improved |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for metric, item in summary.items():
        lines.append(f"| {metric} | {item['d0_mean']:.6f} | {item['d0_worst']:.6f} | {item['d0_std_ddof0']:.6f} | {item['candidate_mean']:.6f} | {item['candidate_worst']:.6f} | {item['candidate_std_ddof0']:.6f} | {item['mean_delta']:+.6f} | {item['improved_folds']}/3 |")
    lines += ["", "## Per-fold delta", "", "| Metric | F1 | F2 | F3 | Mean |", "|---|---:|---:|---:|---:|"]
    for metric, item in summary.items():
        lines.append(f"| {metric} | {item['folds']['F1']['delta']:+.6f} | {item['folds']['F2']['delta']:+.6f} | {item['folds']['F3']['delta']:+.6f} | {item['mean_delta']:+.6f} |")
    lines += ["", "## Complementarity", "", "| Diagnostic | F1 | F2 | F3 | Mean |", "|---|---:|---:|---:|---:|"]
    for metric, item in comp_summary.items():
        lines.append(f"| {metric} | {item['folds']['F1']:.6f} | {item['folds']['F2']:.6f} | {item['folds']['F3']:.6f} | {item['mean_of_fold_means']:.6f} |")
    lines += ["", f"- Stronger main model: `{stronger_main}`", f"- Second model candidate: `{second_model}`",
              f"- Decision: `{payload['decision']}`", ""]
    (out / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    tracker = start_swanlab(root, "D4_summary", stage["swanlab"]["group"],
                            {"phase": "D4", "kind": "three_fold_summary", "folds": stage["folds"],
                             "decision": payload["decision"]},
                            description="D4 three-fold aggregate metrics only")
    tracker.log({"summary": {metric: {"mean": item["candidate_mean"], "worst": item["candidate_worst"],
                                       "std": item["candidate_std_ddof0"], "delta_vs_d0": item["mean_delta"]}
                                      for metric, item in summary.items()},
                 "complementarity": {metric: item["mean_of_fold_means"] for metric, item in comp_summary.items()}})
    tracker.finish()
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", choices=["F1", "F2", "F3"])
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    root = Path.cwd()
    stage = yaml.safe_load((root / "config/phase_d4_lightgbm_rank_regression.yaml").read_text(encoding="utf-8"))
    if args.summarize:
        summarize(root, stage)
    elif args.fold:
        run_fold(root, args.fold, stage)
    else:
        for fold in stage["folds"]:
            run_fold(root, fold, stage)
        summarize(root, stage)


if __name__ == "__main__":
    main()
