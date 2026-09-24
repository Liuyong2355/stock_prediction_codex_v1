"""D3: D0 Candidate trained on only the two calendar years before validation."""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .baselines import configs, peak_memory_gb, save_json
from .build_basic40 import sha256
from .compare_official import load_official, replay
from .e003 import official_daily_and_diagnostic
from .folds import split_fold
from .phase_d0_e006_rank_view import materialize_augmented, resolve_rank_sources
from .phase_d2b_top10_binary import complementarity
from .ranking import iterations, make_model, predict_batches, training_plan


OFFICIAL_METRICS = ("ic_mean", "annual_excess", "mean_turnover", "final_score")
REPORT_METRICS = OFFICIAL_METRICS + ("diagnostic_turnover", "missing_top_fraction")


def recent_training_indices(train, dates, window):
    train = np.asarray(train, dtype=np.int64)
    dates = np.asarray(dates)
    selected = train[(dates[train] >= int(window["train_start"])) & (dates[train] <= int(window["train_end"]))]
    if not len(selected):
        raise ValueError("Recent training window is empty")
    selected_dates = dates[selected]
    if selected_dates.min() < int(window["train_start"]) or selected_dates.max() > int(window["train_end"]):
        raise AssertionError("Recent window boundary violation")
    return selected


def evaluate_frame(root, frame):
    with tempfile.TemporaryDirectory(prefix="phase_d3_official_") as temp:
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
    full_train, valid, split = split_fold(keys["trade_date"], labels, fold, fold_config["purge_rule"]["purge_trading_days"])
    baseline = json.loads((root / "outputs/phase_d0_e006_rank_view" / fold_id / "result.json").read_text(encoding="utf-8"))
    if split != baseline["split"] or len(valid) != baseline["valid_rows"]:
        raise AssertionError("D0 split or validation rows changed")
    window = stage["folds"][fold_id]
    train = recent_training_indices(full_train, keys["trade_date"], window)
    unique_dates = np.unique(keys["trade_date"][train])
    if unique_dates[-1] != split["train_end"] or unique_dates[-1] in split["purge_dates"]:
        raise AssertionError("Frozen purge was not retained")
    order, fit_y, group, _ = training_plan(keys["trade_date"][train], labels[train], "E006")
    if group is not None or not np.array_equal(order, np.arange(len(train))):
        raise AssertionError("E006 rank target/order changed")

    out = root / "outputs/phase_d3_recent_2y" / fold_id
    out.mkdir(parents=True, exist_ok=True)
    if (out / "result.json").exists():
        raise FileExistsError(f"Completed fold exists: {fold_id}")
    started = time.perf_counter()
    train_path, valid_path = out / "training_work.npy", out / "validation_work.npy"
    training = materialize_augmented(matrix, keys, train, source_columns, train_path)
    validation = materialize_augmented(matrix, keys, valid, source_columns, valid_path)
    all_names = names + new_names
    model = make_model(experiments, "E006")
    np.random.seed(experiments["seed"])
    fit_started = time.perf_counter()
    model.fit(pd.DataFrame(training, columns=all_names, copy=False), fit_y)
    fit_seconds = time.perf_counter() - fit_started
    if iterations(model) != 800:
        raise AssertionError("Frozen E006 iteration count changed")
    pred = predict_batches(model, validation, np.arange(len(valid)), all_names)
    if not np.isfinite(pred).all():
        raise AssertionError("Non-finite D3 predictions")
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
        "phase": "D3", "fold": fold_id, "baseline": "D0 Candidate",
        "single_variable": stage["single_variable"], "feature_view": stage["feature_view"],
        "feature_count": len(all_names), "model": "xgboost_reg_v1", "target": "rank",
        "recent_window": window, "full_d0_train_rows": len(full_train), "train_rows": len(train),
        "train_dates": len(unique_dates), "actual_train_start": int(unique_dates[0]),
        "actual_train_end": int(unique_dates[-1]), "valid_rows": len(valid), "split": split,
        "official": official, "diagnostic": diagnostic, "complementarity": comp,
        "baseline_official": baseline["official"], "baseline_diagnostic": baseline["diagnostic"],
        "delta_candidate_minus_d0": {key: float(official[key] - baseline["official"][key]) for key in OFFICIAL_METRICS},
        "fit_seconds": fit_seconds, "elapsed_seconds": time.perf_counter() - started,
        "peak_memory_gb": peak_memory_gb(), "seed": experiments["seed"], "postprocess": None,
        "config_sha256": sha256(root / "config/phase_d3_recent_2y.yaml"),
        "feature_config_sha256": sha256(root / stage["feature_config"]),
        "official_evaluator_sha256": sha256(root / "evaluate.py"),
    }
    save_json(out / "result.json", result)
    return result


def summarize(root: Path, stage: dict) -> dict:
    rows = [json.loads((root / "outputs/phase_d3_recent_2y" / f / "result.json").read_text(encoding="utf-8")) for f in stage["folds"]]
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
                                "highest_fold": float(fold_means.max()),
                                "std_across_folds_ddof0": float(fold_means.std(ddof=0)),
                                "folds": {row["fold"]: float(v) for row, v in zip(rows, fold_means)}}
    stronger_main = (summary["final_score"]["improved_folds"] >= 2 and summary["final_score"]["mean_delta"] > 0
                     and summary["final_score"]["folds"]["F2"]["delta"] >= 0)
    second_alpha = (min(summary["ic_mean"]["folds"][f]["candidate"] for f in stage["folds"]) > 0
                    and min(summary["annual_excess"]["folds"][f]["candidate"] for f in stage["folds"]) > 0
                    and summary["final_score"]["candidate_worst"] > 0)
    payload = {"phase": "D3", "comparison": "D0 expanding history vs Recent-2Y",
               "fold_results": rows, "summary": summary, "complementarity": comp_summary,
               "stronger_main_model": stronger_main, "second_alpha_candidate": second_alpha,
               "decision": "retain_recent_2y_without_fusion" if (stronger_main or second_alpha) else "stop_training_window_route"}
    out = root / "outputs/phase_d3_recent_2y"
    save_json(out / "comparison.json", payload)
    lines = ["# Phase D3：Recent-2Y training window", "",
             "唯一变量：训练历史由 D0 expanding history 改为验证年前最近两个自然年；其余条件不变。无调参、postprocess 或融合。", "",
             "| Metric | D0 mean | D0 worst | D0 std | Recent-2Y mean | Recent-2Y worst | Recent-2Y std | Mean Δ | Improved |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for metric, item in summary.items():
        lines.append(f"| {metric} | {item['d0_mean']:.6f} | {item['d0_worst']:.6f} | {item['d0_std_ddof0']:.6f} | {item['candidate_mean']:.6f} | {item['candidate_worst']:.6f} | {item['candidate_std_ddof0']:.6f} | {item['mean_delta']:+.6f} | {item['improved_folds']}/3 |")
    lines += ["", "## Per-fold delta", "", "| Metric | F1 | F2 | F3 | Mean |", "|---|---:|---:|---:|---:|"]
    for metric, item in summary.items():
        lines.append(f"| {metric} | {item['folds']['F1']['delta']:+.6f} | {item['folds']['F2']['delta']:+.6f} | {item['folds']['F3']['delta']:+.6f} | {item['mean_delta']:+.6f} |")
    lines += ["", "## Complementarity", "", "| Diagnostic | F1 | F2 | F3 | Mean |", "|---|---:|---:|---:|---:|"]
    for metric, item in comp_summary.items():
        lines.append(f"| {metric} | {item['folds']['F1']:.6f} | {item['folds']['F2']:.6f} | {item['folds']['F3']:.6f} | {item['mean_of_fold_means']:.6f} |")
    lines += ["", f"- Stronger main model: `{stronger_main}`", f"- Second alpha candidate: `{second_alpha}`",
              f"- Decision: `{payload['decision']}`", ""]
    (out / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", choices=["F1", "F2", "F3"])
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    root = Path.cwd()
    stage = yaml.safe_load((root / "config/phase_d3_recent_2y.yaml").read_text(encoding="utf-8"))
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
