"""Conditional D1: apply the successful D0 rank view to corrected LambdaRank."""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from .baselines import configs, peak_memory_gb, save_json
from .build_basic40 import sha256
from .compare_official import load_official
from .folds import split_fold
from .phase_c2 import evaluate_prediction, make_model, relevance_and_groups
from .phase_d0_e006_rank_view import materialize_augmented, resolve_rank_sources


METRICS = ("ic_mean", "annual_excess", "mean_turnover", "final_score")


def baseline_rows(root: Path) -> dict[str, dict]:
    payload = json.loads((root / "outputs/phase_c2/phase_c2_comparison.json").read_text(encoding="utf-8"))
    rows = [row for row in payload["results"] if row["candidate"] == "C2-R1" and row["variant"] == "raw"]
    if {row["fold"] for row in rows} != {"F1", "F2", "F3"}:
        raise ValueError("Incomplete corrected LambdaRank baseline")
    return {row["fold"]: row for row in rows}


def run_fold(root: Path, fold_id: str, view: dict, ranker: dict) -> dict:
    fold_config, _ = configs(root)
    manifest = json.loads((root / "outputs/full147_manifest.json").read_text(encoding="utf-8"))
    names = list(manifest["feature_names"])
    sources, new_names = resolve_rank_sources(view, names)
    columns = [names.index(name) for name in sources]
    matrix = np.load(manifest["datasets"]["train"]["files"]["matrix"]["path"], mmap_mode="r")
    keys = np.load(manifest["datasets"]["train"]["files"]["keys"]["path"], mmap_mode="r")
    labels = np.load(root / "outputs/baselines/cache/labels.npy", mmap_mode="r")
    fold = next(item for item in fold_config["folds"] if item["id"] == fold_id)
    train, valid, split = split_fold(keys["trade_date"], labels, fold, fold_config["purge_rule"]["purge_trading_days"])
    order, fit_y, groups, _ = relevance_and_groups(keys["trade_date"][train], labels[train])
    fit_indices = train[order]
    out = root / "outputs/phase_d1_lambdarank_rank_view" / fold_id
    out.mkdir(parents=True, exist_ok=True)
    if (out / "result.json").exists():
        raise FileExistsError(f"Completed fold exists: {fold_id}")
    train_path, valid_path = out / "training_work.npy", out / "validation_work.npy"
    started = time.perf_counter()
    training = materialize_augmented(matrix, keys, fit_indices, columns, train_path)
    validation = materialize_augmented(matrix, keys, valid, columns, valid_path)
    feature_names = names + new_names
    model = make_model(ranker["ranker"])
    fit_started = time.perf_counter()
    model.fit(pd.DataFrame(training, columns=feature_names, copy=False), fit_y, group=groups)
    fit_seconds = time.perf_counter() - fit_started
    if model.booster_.current_iteration() != 100:
        raise AssertionError("Corrected LambdaRank iterations changed")
    pred = model.predict(pd.DataFrame(validation, columns=feature_names, copy=False))
    if not np.isfinite(pred).all():
        raise AssertionError("Non-finite predictions")
    control = pd.read_csv(root / "outputs/baselines/E005" / fold_id / "predictions.csv.gz", float_precision="round_trip")
    with tempfile.TemporaryDirectory(prefix="phase_d1_official_") as temp:
        official, diagnostic, frame = evaluate_prediction(root, pred, control, load_official(root), temp)
    frame.to_csv(out / "predictions.csv.gz", index=False, float_format="%.17g", compression={"method": "gzip", "mtime": 0})
    joblib.dump({"model": model, "feature_names": feature_names}, out / "model.joblib")
    for item in (training, validation):
        item._mmap.close()
    train_path.unlink(); valid_path.unlink()
    baseline = baseline_rows(root)[fold_id]
    delta = {metric: float(official[metric] - baseline[metric]) for metric in METRICS}
    result = {
        "phase": "D1", "fold": fold_id, "baseline": "C2-R1 raw corrected LambdaRank",
        "feature_view": view["candidate_feature_view"], "base_feature_count": len(names),
        "new_rank_feature_count": len(new_names), "new_rank_sources": sources,
        "model": "corrected_lambdarank", "postprocess": None, "split": split,
        "train_rows": len(train), "fit_rows": len(fit_y), "valid_rows": len(valid),
        "groups": len(groups), "official": official, "diagnostic": diagnostic,
        "baseline_metrics": baseline, "delta_candidate_minus_c2_r1": delta,
        "fit_seconds": fit_seconds, "elapsed_seconds": time.perf_counter() - started,
        "peak_memory_gb": peak_memory_gb(),
        "view_config_sha256": sha256(root / "config/phase_d0_e006_rank_view.yaml"),
        "ranker_config_sha256": sha256(root / "config/phase_c2_lambdarank.yaml"),
        "official_evaluator_sha256": sha256(root / "evaluate.py"),
    }
    save_json(out / "result.json", result)
    return result


def summarize(root: Path) -> dict:
    rows = [json.loads((root / "outputs/phase_d1_lambdarank_rank_view" / fold / "result.json").read_text(encoding="utf-8")) for fold in ("F1", "F2", "F3")]
    summary = {}
    for metric in METRICS:
        candidate = np.array([row["official"][metric] for row in rows])
        baseline = np.array([row["baseline_metrics"][metric] for row in rows])
        better = candidate < baseline if metric == "mean_turnover" else candidate > baseline
        summary[metric] = {"baseline_mean": float(baseline.mean()), "candidate_mean": float(candidate.mean()),
                           "mean_delta": float((candidate - baseline).mean()), "improved_folds": int(better.sum()),
                           "candidate_std_ddof0": float(candidate.std(ddof=0))}
    missing = np.array([row["diagnostic"]["missing_top_fraction"] for row in rows])
    baseline_missing = np.array([row["baseline_metrics"]["missing_top_fraction"] for row in rows])
    signal_ok = ((summary["ic_mean"]["mean_delta"] > 0 and summary["ic_mean"]["improved_folds"] >= 2) or
                 (summary["annual_excess"]["mean_delta"] > 0 and summary["annual_excess"]["improved_folds"] >= 2))
    decision = "keep_rank_view_for_both_models" if signal_ok and missing.mean() <= baseline_missing.mean() + .01 else "e006_only_do_not_generalize"
    payload = {"phase": "D1", "comparison": "C2-R1 raw vs C2-R1 raw plus D0 rank view",
               "folds": rows, "summary": summary, "mean_missing_top_fraction": float(missing.mean()), "decision": decision}
    out = root / "outputs/phase_d1_lambdarank_rank_view"; save_json(out / "comparison.json", payload)
    lines = ["# Phase D1：corrected LambdaRank feature representation", "", "| Metric | C2-R1 | Candidate | Δ | Improved folds | Candidate std |", "|---|---:|---:|---:|---:|---:|"]
    for metric, item in summary.items():
        lines.append(f"| {metric} | {item['baseline_mean']:.6f} | {item['candidate_mean']:.6f} | {item['mean_delta']:+.6f} | {item['improved_folds']}/3 | {item['candidate_std_ddof0']:.6f} |")
    lines += ["", f"- Mean missing-label Top fraction: {payload['mean_missing_top_fraction']:.2%}", f"- Decision: `{decision}`", ""]
    (out / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", choices=["F1", "F2", "F3"])
    args = parser.parse_args(); root = Path.cwd()
    view = yaml.safe_load((root / "config/phase_d0_e006_rank_view.yaml").read_text(encoding="utf-8"))
    ranker = yaml.safe_load((root / "config/phase_c2_lambdarank.yaml").read_text(encoding="utf-8"))
    if args.fold:
        run_fold(root, args.fold, view, ranker)
    else:
        for fold in ("F1", "F2", "F3"):
            run_fold(root, fold, view, ranker)
        summarize(root)


if __name__ == "__main__":
    main()
