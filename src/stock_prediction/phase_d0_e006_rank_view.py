"""E006 single-variable feature-view experiment: add mapped same-date ranks."""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .baselines import configs, materialize, peak_memory_gb, save_json
from .build_basic40 import sha256
from .compare_official import load_official, replay
from .e003 import official_daily_and_diagnostic
from .folds import split_fold
from .ranking import iterations, make_model, predict_batches, training_plan


METRICS = ("ic_mean", "annual_excess", "mean_turnover", "final_score")


def resolve_rank_sources(config: dict, full_names: list[str]) -> tuple[list[str], list[str]]:
    sources = list(config["source_feature_mapping"])
    if len(sources) != len(set(sources)) or not set(sources) <= set(full_names):
        raise ValueError("Rank source allowlist must be unique Full147 features")
    existing = dict(config["existing_rank_dependencies"])
    if not set(existing) <= set(sources):
        raise ValueError("Existing-rank dependency keys must be mapped sources")
    for source, rank_name in existing.items():
        if rank_name not in full_names:
            raise ValueError(f"Missing existing rank feature: {rank_name}")
    new_sources = [name for name in sources if name not in existing]
    new_names = [f"{name}{config['new_feature_suffix']}" for name in new_sources]
    if set(new_names) & set(full_names):
        raise ValueError("New rank names collide with Full147")
    return new_sources, new_names


def rank_block(values: np.ndarray) -> np.ndarray:
    """Column-wise percentile ranks; average ties and finite values only."""
    frame = pd.DataFrame(values, copy=False)
    return frame.rank(axis=0, method="average", pct=True, na_option="keep").to_numpy(dtype=np.float64)


def materialize_augmented(matrix, keys, indices, source_columns, destination: Path):
    """Write Full147 plus same-date ranks without holding a fold copy in RAM."""
    indices = np.asarray(indices, dtype=np.int64)
    dates = np.asarray(keys["trade_date"][indices])
    if np.any(dates[1:] < dates[:-1]):
        order = np.argsort(dates, kind="stable")
    else:
        order = np.arange(len(indices))
    sorted_indices, sorted_dates = indices[order], dates[order]
    out = np.lib.format.open_memmap(
        destination, mode="w+", dtype=np.float64,
        shape=(len(indices), matrix.shape[1] + len(source_columns)),
    )
    starts = np.r_[0, np.flatnonzero(sorted_dates[1:] != sorted_dates[:-1]) + 1]
    ends = np.r_[starts[1:], len(indices)]
    for start, end in zip(starts, ends):
        target_rows = order[start:end]
        raw_rows = sorted_indices[start:end]
        out[target_rows, : matrix.shape[1]] = matrix[raw_rows]
        values = np.asarray(matrix[np.ix_(raw_rows, source_columns)], dtype=np.float64)
        out[target_rows, matrix.shape[1] :] = rank_block(values)
    out.flush()
    return out


def evaluate_frame(root: Path, frame: pd.DataFrame) -> tuple[dict, dict]:
    with tempfile.TemporaryDirectory(prefix="phase_d0_official_") as temp:
        official = replay(frame, temp, load_official(root))
    _, diagnostic, _, _ = official_daily_and_diagnostic(frame)
    return official, diagnostic


def run_fold(root: Path, fold_id: str, stage: dict) -> dict:
    fold_config, experiments = configs(root)
    manifest = json.loads((root / "outputs/full147_manifest.json").read_text(encoding="utf-8"))
    full_names = list(manifest["feature_names"])
    new_sources, new_names = resolve_rank_sources(stage, full_names)
    source_columns = [full_names.index(name) for name in new_sources]
    matrix = np.load(manifest["datasets"]["train"]["files"]["matrix"]["path"], mmap_mode="r")
    keys = np.load(manifest["datasets"]["train"]["files"]["keys"]["path"], mmap_mode="r")
    labels = np.load(root / "outputs/baselines/cache/labels.npy", mmap_mode="r")
    fold = next(item for item in fold_config["folds"] if item["id"] == fold_id)
    train, valid, split = split_fold(keys["trade_date"], labels, fold, fold_config["purge_rule"]["purge_trading_days"])
    baseline = json.loads((root / "outputs/baselines/E006" / fold_id / "result.json").read_text(encoding="utf-8"))
    baseline_diagnostic = json.loads((root / "outputs/baselines/E006" / fold_id / "missing_label_diagnostic.json").read_text(encoding="utf-8"))
    if split != baseline["split"]:
        raise AssertionError("E006 split changed")
    order, fit_y, group, _ = training_plan(keys["trade_date"][train], labels[train], "E006")
    if group is not None or not np.array_equal(order, np.arange(len(train))):
        raise AssertionError("E006 target/order changed")
    out = root / "outputs/phase_d0_e006_rank_view" / fold_id
    out.mkdir(parents=True, exist_ok=True)
    if (out / "result.json").exists():
        raise FileExistsError(f"Completed fold exists: {fold_id}")
    started = time.perf_counter()
    train_path, valid_path = out / "training_work.npy", out / "validation_work.npy"
    training = materialize_augmented(matrix, keys, train, source_columns, train_path)
    validation = materialize_augmented(matrix, keys, valid, source_columns, valid_path)
    names = full_names + new_names
    model = make_model(experiments, "E006")
    np.random.seed(experiments["seed"])
    fit_started = time.perf_counter()
    model.fit(pd.DataFrame(training, columns=names, copy=False), fit_y)
    fit_seconds = time.perf_counter() - fit_started
    if iterations(model) != 800:
        raise AssertionError("Frozen E006 iteration count changed")
    pred = predict_batches(model, validation, np.arange(len(valid)), names)
    if not np.isfinite(pred).all():
        raise AssertionError("Non-finite candidate predictions")
    frame = pd.DataFrame({
        "ts_code": keys["ts_code"][valid], "trade_date": keys["trade_date"][valid],
        "pred": pred, "y_ret_1d": np.asarray(labels[valid]),
        "flag_limit_up": matrix[valid, full_names.index("flag_limit_up")],
    })
    frame.to_csv(out / "predictions.csv.gz", index=False, float_format="%.17g", compression={"method": "gzip", "mtime": 0})
    official, diagnostic = evaluate_frame(root, frame)
    model.save_model(out / "model.ubj")
    for item in (training, validation):
        item._mmap.close()
    train_path.unlink(); valid_path.unlink()
    baseline_metrics = baseline["official"]
    delta = {key: float(official[key] - baseline_metrics[key]) for key in METRICS}
    result = {
        "phase": "D0", "fold": fold_id, "baseline": "E006",
        "feature_view": stage["candidate_feature_view"], "base_feature_count": len(full_names),
        "mapped_source_count": len(stage["source_feature_mapping"]),
        "existing_rank_count": len(stage["existing_rank_dependencies"]),
        "new_rank_feature_count": len(new_names), "new_rank_sources": new_sources,
        "new_feature_names": new_names, "model": "xgboost_reg_v1", "target": "rank",
        "split": split, "train_rows": len(train), "valid_rows": len(valid),
        "official": official, "diagnostic": diagnostic, "baseline_official": baseline_metrics,
        "baseline_diagnostic": baseline_diagnostic,
        "delta_candidate_minus_e006": delta, "fit_seconds": fit_seconds,
        "elapsed_seconds": time.perf_counter() - started, "peak_memory_gb": peak_memory_gb(),
        "config_sha256": sha256(root / "config/phase_d0_e006_rank_view.yaml"),
        "full147_manifest_sha256": sha256(root / "outputs/full147_manifest.json"),
        "official_evaluator_sha256": sha256(root / "evaluate.py"),
        "postprocess": None,
    }
    save_json(out / "result.json", result)
    return result


def summarize(root: Path, stage: dict) -> dict:
    rows = [json.loads((root / "outputs/phase_d0_e006_rank_view" / fold / "result.json").read_text(encoding="utf-8")) for fold in stage["folds"]]
    summary = {}
    for metric in METRICS:
        candidate = np.array([row["official"][metric] for row in rows])
        baseline = np.array([row["baseline_official"][metric] for row in rows])
        summary[metric] = {
            "baseline_mean": float(baseline.mean()), "candidate_mean": float(candidate.mean()),
            "mean_delta": float((candidate - baseline).mean()),
            "improved_folds": int(((candidate > baseline) if metric != "mean_turnover" else (candidate < baseline)).sum()),
            "candidate_std_ddof0": float(candidate.std(ddof=0)),
        }
    missing = [row["diagnostic"]["missing_share_of_top"] for row in rows]
    diagnostic_turnover = [row["diagnostic"]["diagnostic_exclude_missing_y_turnover"] for row in rows]
    ic_ok = summary["ic_mean"]["mean_delta"] > 0 and summary["ic_mean"]["improved_folds"] >= 2
    excess_ok = summary["annual_excess"]["mean_delta"] > 0 and summary["annual_excess"]["improved_folds"] >= 2
    baseline_missing = np.mean([row["baseline_diagnostic"]["missing_share_of_top"] for row in rows])
    decision = "continue_to_corrected_lambdarank" if (ic_ok or excess_ok) and np.mean(missing) <= baseline_missing + .01 else "stop_teammate_feature_view"
    payload = {"phase": "D0", "comparison": "E006 Full147 vs Full147 plus mapped core ranks", "folds": rows,
               "summary": summary, "mean_diagnostic_turnover": float(np.mean(diagnostic_turnover)),
               "mean_missing_top_fraction": float(np.mean(missing)), "decision": decision}
    out = root / "outputs/phase_d0_e006_rank_view"; save_json(out / "comparison.json", payload)
    lines = ["# Phase D0：E006 feature representation", "", "| Metric | E006 | Candidate | Δ | Improved folds | Candidate std |", "|---|---:|---:|---:|---:|---:|"]
    for metric, item in summary.items():
        lines.append(f"| {metric} | {item['baseline_mean']:.6f} | {item['candidate_mean']:.6f} | {item['mean_delta']:+.6f} | {item['improved_folds']}/3 | {item['candidate_std_ddof0']:.6f} |")
    lines += ["", f"- Mean diagnostic turnover: {payload['mean_diagnostic_turnover']:.6f}",
              f"- Mean missing-label Top fraction: {payload['mean_missing_top_fraction']:.2%}",
              f"- Decision: `{decision}`", ""]
    (out / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", choices=["F1", "F2", "F3"])
    args = parser.parse_args()
    root = Path.cwd()
    stage = yaml.safe_load((root / "config/phase_d0_e006_rank_view.yaml").read_text(encoding="utf-8"))
    if args.fold:
        run_fold(root, args.fold, stage)
    else:
        for fold in stage["folds"]:
            run_fold(root, fold, stage)
        summarize(root, stage)


if __name__ == "__main__":
    main()
