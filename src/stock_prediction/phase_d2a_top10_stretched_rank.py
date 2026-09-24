"""D2-A: D0 Candidate with only the continuous rank training target changed."""
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
from .ranking import iterations, make_model, predict_batches


OFFICIAL_METRICS = ("ic_mean", "annual_excess", "mean_turnover", "final_score")
REPORT_METRICS = OFFICIAL_METRICS + ("diagnostic_turnover", "missing_top_fraction")


def top10_stretched_rank_target(dates, labels, threshold=0.9, coefficient=10.0):
    """Daily finite-label average percentile rank with only its top decile stretched."""
    dates = np.asarray(dates)
    labels = np.asarray(labels, dtype=np.float64)
    if dates.ndim != 1 or labels.ndim != 1 or len(dates) != len(labels) or not len(labels):
        raise ValueError("Aligned nonempty training vectors required")
    if not np.isfinite(labels).all():
        raise ValueError("Pass only finite supervised labels after purge/filtering")
    if threshold != 0.9 or coefficient != 10.0:
        raise ValueError("D2-A fixes threshold=0.9 and coefficient=10")
    u = pd.Series(labels).groupby(dates, sort=False).rank(method="average", pct=True).to_numpy()
    transformed = np.where(u <= threshold, u, threshold + coefficient * (u - threshold))
    return transformed - 0.5


def evaluate_frame(root, frame):
    with tempfile.TemporaryDirectory(prefix="phase_d2a_official_") as temp:
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
    if split != baseline["split"]:
        raise AssertionError("D0 split changed")
    fit_y = top10_stretched_rank_target(keys["trade_date"][train], labels[train])
    ordinary_u = pd.Series(np.asarray(labels[train])).groupby(np.asarray(keys["trade_date"][train]), sort=False).rank(method="average", pct=True).to_numpy()
    ordinary_target = ordinary_u - 0.5
    np.testing.assert_array_equal(fit_y[ordinary_u <= 0.9], ordinary_target[ordinary_u <= 0.9])
    if not np.all(fit_y[ordinary_u > 0.9] > ordinary_target[ordinary_u > 0.9]):
        raise AssertionError("Top-decile distances were not stretched")

    out = root / "outputs/phase_d2a_top10_stretched_rank" / fold_id
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
        raise AssertionError("Non-finite D2-A predictions")
    frame = pd.DataFrame({"ts_code": keys["ts_code"][valid], "trade_date": keys["trade_date"][valid],
                          "pred": pred, "y_ret_1d": np.asarray(labels[valid]),
                          "flag_limit_up": matrix[valid, names.index("flag_limit_up")]})
    frame.to_csv(out / "predictions.csv.gz", index=False, float_format="%.17g", compression={"method": "gzip", "mtime": 0})
    official, diagnostic = evaluate_frame(root, frame)
    model.save_model(out / "model.ubj")
    for item in (training, validation):
        item._mmap.close()
    train_path.unlink(); valid_path.unlink()
    baseline_values = baseline["official"]
    result = {
        "phase": "D2-A", "fold": fold_id, "baseline": "D0 Candidate",
        "feature_view": stage["feature_view"], "feature_count": len(all_names),
        "model": "xgboost_reg_v1", "target": stage["target"], "split": split,
        "train_rows": len(train), "valid_rows": len(valid), "official": official,
        "diagnostic": diagnostic, "baseline_official": baseline_values,
        "baseline_diagnostic": baseline["diagnostic"],
        "delta_candidate_minus_d0": {key: float(official[key] - baseline_values[key]) for key in OFFICIAL_METRICS},
        "fit_seconds": fit_seconds, "elapsed_seconds": time.perf_counter() - started,
        "peak_memory_gb": peak_memory_gb(), "seed": experiments["seed"], "postprocess": None,
        "config_sha256": sha256(root / "config/phase_d2a_top10_stretched_rank.yaml"),
        "feature_config_sha256": sha256(root / stage["feature_config"]),
        "official_evaluator_sha256": sha256(root / "evaluate.py"),
    }
    save_json(out / "result.json", result)
    return result


def summarize(root: Path, stage: dict) -> dict:
    rows = [json.loads((root / "outputs/phase_d2a_top10_stretched_rank" / f / "result.json").read_text(encoding="utf-8")) for f in stage["folds"]]
    def values(row, metric, baseline=False):
        if metric == "diagnostic_turnover":
            return row["baseline_diagnostic" if baseline else "diagnostic"]["diagnostic_exclude_missing_y_turnover"]
        if metric == "missing_top_fraction":
            return row["baseline_diagnostic" if baseline else "diagnostic"]["missing_share_of_top"]
        return row["baseline_official" if baseline else "official"][metric]
    summary = {}
    for metric in REPORT_METRICS:
        candidate = np.array([values(row, metric) for row in rows], dtype=float)
        baseline = np.array([values(row, metric, True) for row in rows], dtype=float)
        lower_better = "turnover" in metric or metric == "missing_top_fraction"
        summary[metric] = {
            "d0_mean": float(baseline.mean()), "candidate_mean": float(candidate.mean()),
            "mean_delta": float((candidate - baseline).mean()),
            "improved_folds": int(((candidate < baseline) if lower_better else (candidate > baseline)).sum()),
            "d0_worst": float(baseline.max() if lower_better else baseline.min()),
            "d0_std_ddof0": float(baseline.std(ddof=0)),
            "candidate_worst": float(candidate.max() if lower_better else candidate.min()),
            "candidate_std_ddof0": float(candidate.std(ddof=0)),
            "folds": {row["fold"]: {"d0": float(b), "candidate": float(c), "delta": float(c-b)} for row, b, c in zip(rows, baseline, candidate)},
        }
    score_delta = summary["final_score"]["mean_delta"]
    decomposition = {
        "ic_contribution": 0.4 * summary["ic_mean"]["mean_delta"],
        "top10_excess_contribution": 0.3 * summary["annual_excess"]["mean_delta"],
        "turnover_contribution": -0.3 * summary["mean_turnover"]["mean_delta"],
    }
    if not np.isclose(sum(decomposition.values()), score_delta, atol=1e-12, rtol=0):
        raise AssertionError("Score decomposition mismatch")
    checks = {
        "top10_excess_improves_2_of_3": summary["annual_excess"]["improved_folds"] >= 2,
        "score_improves_2_of_3": summary["final_score"]["improved_folds"] >= 2,
        "f2_score_non_degraded": summary["final_score"]["folds"]["F2"]["delta"] >= 0,
        "rank_ic_basic_stability": abs(summary["ic_mean"]["mean_delta"]) <= 0.005,
        "mean_score_at_least_0_250": summary["final_score"]["candidate_mean"] >= stage["breakthrough_threshold_mean_score"],
    }
    breakthrough = all(checks.values())
    payload = {"phase": "D2-A", "comparison": "D0 Candidate vs Top10-stretched continuous rank target",
               "fold_results": rows, "summary": summary, "score_delta_decomposition": decomposition,
               "decision_checks": checks, "structural_breakthrough_candidate": breakthrough}
    out = root / "outputs/phase_d2a_top10_stretched_rank"
    save_json(out / "comparison.json", payload)
    lines = ["# Phase D2-A：Top10-stretched continuous rank target", "",
             "唯一变量：训练目标由 `u-0.5` 改为 `T(u)-0.5`；特征、模型参数、fold、purge、seed、evaluator 与无 postprocess 均保持 D0 不变。", "",
             "| Metric | D0 mean | D0 worst | D0 std | D2-A mean | D2-A worst | D2-A std | Mean Δ | Improved |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for metric, item in summary.items():
        lines.append(f"| {metric} | {item['d0_mean']:.6f} | {item['d0_worst']:.6f} | {item['d0_std_ddof0']:.6f} | {item['candidate_mean']:.6f} | {item['candidate_worst']:.6f} | {item['candidate_std_ddof0']:.6f} | {item['mean_delta']:+.6f} | {item['improved_folds']}/3 |")
    lines += ["", "## Per-fold delta", "", "| Metric | F1 | F2 | F3 | Mean |", "|---|---:|---:|---:|---:|"]
    for metric, item in summary.items():
        lines.append(f"| {metric} | {item['folds']['F1']['delta']:+.6f} | {item['folds']['F2']['delta']:+.6f} | {item['folds']['F3']['delta']:+.6f} | {item['mean_delta']:+.6f} |")
    lines += ["", "## Score delta decomposition", "",
              f"- IC: {decomposition['ic_contribution']:+.6f}",
              f"- Top10% excess: {decomposition['top10_excess_contribution']:+.6f}",
              f"- Official turnover: {decomposition['turnover_contribution']:+.6f}",
              f"- Structural breakthrough candidate: `{breakthrough}`", ""]
    (out / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", choices=["F1", "F2", "F3"])
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    root = Path.cwd()
    stage = yaml.safe_load((root / "config/phase_d2a_top10_stretched_rank.yaml").read_text(encoding="utf-8"))
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
