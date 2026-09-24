"""D2-B: D0 Candidate features with a fixed Top10% binary training task."""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

from .baselines import configs, peak_memory_gb, save_json
from .build_basic40 import sha256
from .compare_official import load_official, replay
from .e003 import official_daily_and_diagnostic
from .folds import split_fold
from .phase_d0_e006_rank_view import materialize_augmented, resolve_rank_sources


OFFICIAL_METRICS = ("ic_mean", "annual_excess", "mean_turnover", "final_score")
REPORT_METRICS = OFFICIAL_METRICS + ("diagnostic_turnover", "missing_top_fraction")


def top10_binary_target(dates, labels, threshold=0.9):
    """Daily finite-label average percentile rank converted to fixed u>0.9 labels."""
    dates = np.asarray(dates)
    labels = np.asarray(labels, dtype=np.float64)
    if dates.ndim != 1 or labels.ndim != 1 or len(dates) != len(labels) or not len(labels):
        raise ValueError("Aligned nonempty training vectors required")
    if not np.isfinite(labels).all():
        raise ValueError("Pass only finite supervised labels after purge/filtering")
    if threshold != 0.9:
        raise ValueError("D2-B fixes the threshold at 0.9")
    u = pd.Series(labels).groupby(dates, sort=False).rank(method="average", pct=True).to_numpy()
    return (u > threshold).astype(np.int8)


def make_binary_model(experiments):
    entry = next(e for e in experiments["experiments"] if e["id"] == "E006")
    if (entry["feature_set"], entry["model"], entry["target"]) != ("Full147", "xgboost_reg_v1", "rank"):
        raise AssertionError("Frozen E006 definition changed")
    params = dict(experiments["model_catalog"]["xgboost_reg_v1"]["params"])
    changed = {"objective": (params["objective"], "binary:logistic")}
    params["objective"] = "binary:logistic"
    return xgb.XGBClassifier(**params), params, changed


def predict_positive_batches(model, matrix, names, batch_size=100_000):
    result = np.empty(len(matrix), dtype=np.float64)
    for start in range(0, len(matrix), batch_size):
        block = matrix[start:start + batch_size]
        result[start:start + len(block)] = model.predict_proba(
            pd.DataFrame(block, columns=names, copy=False)
        )[:, 1]
    return result


def evaluate_frame(root, frame):
    with tempfile.TemporaryDirectory(prefix="phase_d2b_official_") as temp:
        official = replay(frame, temp, load_official(root))
    _, diagnostic, _, _ = official_daily_and_diagnostic(frame)
    return official, diagnostic


def complementarity(candidate, baseline):
    keys = ["ts_code", "trade_date"]
    merged = candidate[keys + ["pred", "flag_limit_up"]].merge(
        baseline[keys + ["pred"]], on=keys, how="inner", validate="one_to_one", suffixes=("_d2b", "_d0")
    )
    if len(merged) != len(candidate) or len(merged) != len(baseline):
        raise AssertionError("D2-B and D0 validation rows differ")
    daily = []
    for date, group in merged.groupby("trade_date", sort=True):
        rho = group["pred_d2b"].corr(group["pred_d0"], method="spearman")
        eligible = group[group["flag_limit_up"] == 0]
        n_top = max(len(eligible) // 10, 1)
        d2b = set(eligible.sort_values("pred_d2b", ascending=False)["ts_code"].iloc[:n_top])
        d0 = set(eligible.sort_values("pred_d0", ascending=False)["ts_code"].iloc[:n_top])
        intersection = len(d2b & d0)
        daily.append({"trade_date": int(date), "prediction_spearman": float(rho),
                      "top10_jaccard": intersection / len(d2b | d0),
                      "top10_overlap": intersection / min(len(d2b), len(d0)),
                      "top_size": n_top, "intersection": intersection})
    frame = pd.DataFrame(daily)
    return {metric: {"mean": float(frame[metric].mean()), "worst": float(frame[metric].max()),
                     "std_ddof0": float(frame[metric].std(ddof=0))}
            for metric in ("prediction_spearman", "top10_jaccard", "top10_overlap")}, frame


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
    if split != baseline["split"]:
        raise AssertionError("D0 split changed")
    fit_y = top10_binary_target(keys["trade_date"][train], labels[train])
    if set(np.unique(fit_y)) != {0, 1}:
        raise AssertionError("Both binary classes are required")

    out = root / "outputs/phase_d2b_top10_binary" / fold_id
    out.mkdir(parents=True, exist_ok=True)
    if (out / "result.json").exists():
        raise FileExistsError(f"Completed fold exists: {fold_id}")
    started = time.perf_counter()
    train_path, valid_path = out / "training_work.npy", out / "validation_work.npy"
    training = materialize_augmented(matrix, keys, train, source_columns, train_path)
    validation = materialize_augmented(matrix, keys, valid, source_columns, valid_path)
    all_names = names + new_names
    model, model_params, changed_params = make_binary_model(experiments)
    np.random.seed(experiments["seed"])
    fit_started = time.perf_counter()
    model.fit(pd.DataFrame(training, columns=all_names, copy=False), fit_y)
    fit_seconds = time.perf_counter() - fit_started
    if model.get_booster().num_boosted_rounds() != 800:
        raise AssertionError("Frozen E006 iteration count changed")
    pred = predict_positive_batches(model, validation, all_names)
    if not np.isfinite(pred).all() or np.any((pred < 0) | (pred > 1)):
        raise AssertionError("Invalid positive-class probabilities")
    frame = pd.DataFrame({"ts_code": keys["ts_code"][valid], "trade_date": keys["trade_date"][valid],
                          "pred": pred, "y_ret_1d": np.asarray(labels[valid]),
                          "flag_limit_up": matrix[valid, names.index("flag_limit_up")]})
    frame.to_csv(out / "predictions.csv.gz", index=False, float_format="%.17g", compression={"method": "gzip", "mtime": 0})
    official, diagnostic = evaluate_frame(root, frame)
    d0_predictions = pd.read_csv(root / "outputs/phase_d0_e006_rank_view" / fold_id / "predictions.csv.gz",
                                 float_precision="round_trip")
    comp, daily_comp = complementarity(frame, d0_predictions)
    daily_comp.to_csv(out / "daily_complementarity.csv", index=False, float_format="%.17g")
    model.save_model(out / "model.ubj")
    for item in (training, validation):
        item._mmap.close()
    train_path.unlink(); valid_path.unlink()
    result = {
        "phase": "D2-B", "fold": fold_id, "baseline": "D0 Candidate",
        "feature_view": stage["feature_view"], "feature_count": len(all_names),
        "model": "xgboost_binary_classifier", "model_params": model_params,
        "parameters_changed_from_e006": changed_params, "target": stage["target"],
        "positive_rows": int(fit_y.sum()), "positive_fraction": float(fit_y.mean()),
        "split": split, "train_rows": len(train), "valid_rows": len(valid),
        "official": official, "diagnostic": diagnostic, "complementarity": comp,
        "baseline_official": baseline["official"], "baseline_diagnostic": baseline["diagnostic"],
        "delta_candidate_minus_d0": {key: float(official[key] - baseline["official"][key]) for key in OFFICIAL_METRICS},
        "fit_seconds": fit_seconds, "elapsed_seconds": time.perf_counter() - started,
        "peak_memory_gb": peak_memory_gb(), "seed": experiments["seed"], "postprocess": None,
        "config_sha256": sha256(root / "config/phase_d2b_top10_binary.yaml"),
        "feature_config_sha256": sha256(root / stage["feature_config"]),
        "official_evaluator_sha256": sha256(root / "evaluate.py"),
    }
    save_json(out / "result.json", result)
    return result


def summarize(root: Path, stage: dict) -> dict:
    rows = [json.loads((root / "outputs/phase_d2b_top10_binary" / f / "result.json").read_text(encoding="utf-8")) for f in stage["folds"]]
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
    comp_summary = {}
    for metric in ("prediction_spearman", "top10_jaccard", "top10_overlap"):
        fold_means = np.array([row["complementarity"][metric]["mean"] for row in rows])
        comp_summary[metric] = {"mean_of_fold_means": float(fold_means.mean()),
                                "worst_highest_fold": float(fold_means.max()),
                                "std_across_folds_ddof0": float(fold_means.std(ddof=0)),
                                "folds": {row["fold"]: float(v) for row, v in zip(rows, fold_means)}}
    checks = {
        "top10_excess_improves_2_of_3": summary["annual_excess"]["improved_folds"] >= 2,
        "mean_top10_excess_positive_delta": summary["annual_excess"]["mean_delta"] > 0,
        "f2_top10_excess_not_materially_degraded": summary["annual_excess"]["folds"]["F2"]["delta"] >= -0.01,
        "missing_top_not_worse_by_1pct": summary["missing_top_fraction"]["mean_delta"] <= 0.01,
    }
    independent_tail_alpha = sum(checks.values()) >= 3 and checks["top10_excess_improves_2_of_3"] and checks["mean_top10_excess_positive_delta"]
    payload = {"phase": "D2-B", "comparison": "D0 Candidate vs Top10 binary classifier",
               "fold_results": rows, "summary": summary, "complementarity": comp_summary,
               "decision_checks": checks, "independent_tail_alpha": independent_tail_alpha,
               "decision": "stop_dedicated_tail_target_route" if not independent_tail_alpha else "tail_signal_found_no_fusion_started"}
    out = root / "outputs/phase_d2b_top10_binary"
    save_json(out / "comparison.json", payload)
    lines = ["# Phase D2-B：Top10% binary target", "",
             "唯一任务变化：使用同日有限标签 average percentile rank 的 `u>0.9` 二分类标签；输出正类概率。无调参、无 postprocess、无融合。", "",
             "| Metric | D0 mean | D0 worst | D0 std | D2-B mean | D2-B worst | D2-B std | Mean Δ | Improved |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for metric, item in summary.items():
        lines.append(f"| {metric} | {item['d0_mean']:.6f} | {item['d0_worst']:.6f} | {item['d0_std_ddof0']:.6f} | {item['candidate_mean']:.6f} | {item['candidate_worst']:.6f} | {item['candidate_std_ddof0']:.6f} | {item['mean_delta']:+.6f} | {item['improved_folds']}/3 |")
    lines += ["", "## Per-fold delta", "", "| Metric | F1 | F2 | F3 | Mean |", "|---|---:|---:|---:|---:|"]
    for metric, item in summary.items():
        lines.append(f"| {metric} | {item['folds']['F1']['delta']:+.6f} | {item['folds']['F2']['delta']:+.6f} | {item['folds']['F3']['delta']:+.6f} | {item['mean_delta']:+.6f} |")
    lines += ["", "## Complementarity", "", "| Diagnostic | F1 | F2 | F3 | Mean |", "|---|---:|---:|---:|---:|"]
    for metric, item in comp_summary.items():
        lines.append(f"| {metric} | {item['folds']['F1']:.6f} | {item['folds']['F2']:.6f} | {item['folds']['F3']:.6f} | {item['mean_of_fold_means']:.6f} |")
    lines += ["", f"- Independent tail alpha: `{independent_tail_alpha}`", f"- Decision: `{payload['decision']}`", ""]
    (out / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", choices=["F1", "F2", "F3"])
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    root = Path.cwd()
    stage = yaml.safe_load((root / "config/phase_d2b_top10_binary.yaml").read_text(encoding="utf-8"))
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
