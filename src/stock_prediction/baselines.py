"""Project orchestration only; core training is sklearn Ridge / lightgbm."""
import argparse
import csv
import ctypes
from ctypes import wintypes
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.linear_model import Ridge
import yaml

from .audit import read_chunks
from .build_basic40 import sha256
from .evaluator import evaluate, POLICY
from .folds import split_fold
from .preprocessing import RidgePreprocessor


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def save_json(path, value):
    Path(path).write_text(json.dumps(clean_json(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def peak_memory_gb():
    if os.name != "nt":
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ["PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                                "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                                                "PagefileUsage", "PeakPagefileUsage"]]
    counter = Counters()
    counter.cb = ctypes.sizeof(counter)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counter), counter.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return counter.PeakWorkingSetSize / 1024 ** 3


def configs(root):
    return tuple(yaml.safe_load((root / "config" / name).read_text(encoding="utf-8"))
                 for name in ["folds_v1.yaml", "experiments_v1.yaml"])


def prepare_inputs(root):
    """Validate hashes once, build an aligned label cache from training only."""
    manifest = json.loads((root / "outputs/basic40_manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((root / "outputs/data_audit_summary.json").read_text(encoding="utf-8"))
    if manifest["data_version"] != audit["data_version"]:
        raise ValueError("Data versions differ")
    for item in manifest["datasets"]["train"]["files"].values():
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError("Basic40 artifact hash mismatch")
    raw_path = root / "data/raw/训练集.csv"
    if sha256(raw_path) != audit["datasets"]["train"]["sha256"]:
        raise ValueError("Raw data hash mismatch")
    folder = root / "outputs/baselines/cache"
    folder.mkdir(parents=True, exist_ok=True)
    keys = np.load(manifest["datasets"]["train"]["files"]["keys"]["path"], mmap_mode="r", allow_pickle=False)
    target = np.lib.format.open_memmap(folder / "labels.npy", mode="w+", dtype="float64", shape=(len(keys),))
    offset = 0
    for chunk in read_chunks(raw_path, True):
        k = keys[offset:offset + len(chunk)]
        if not np.array_equal(k["ts_code"], chunk.ts_code.astype(str)) or not np.array_equal(k["trade_date"], chunk.trade_date):
            raise ValueError("Raw labels and saved feature keys are misaligned")
        target[offset:offset + len(chunk)] = chunk.y_ret_1d
        offset += len(chunk)
    if offset != len(keys):
        raise ValueError("Incomplete labels")
    target.flush()
    target._mmap.close()
    save_json(root / "outputs/baselines/input_verification.json", {
        "data_version": manifest["data_version"], "rows": offset, "raw_sha256": audit["datasets"]["train"]["sha256"],
        "features_and_keys_hash_verified": True, "row_alignment_verified": True,
        "labels_sha256": sha256(folder / "labels.npy"), "test_data_used": False})


def make_model(experiment_id, config):
    if experiment_id == "E001":
        return Ridge(**config["model_catalog"]["ridge_v1"]["params"])
    if experiment_id == "E002":
        return lgb.LGBMRegressor(**config["model_catalog"]["lightgbm_reg_v1"]["params"])
    raise ValueError("Only the frozen E001/E002 models are allowed")


def save_lightgbm_text(model, path):
    # Native Windows file APIs can reject Unicode paths. Keep official model
    # serialization, but let Python handle the Unicode filename.
    Path(path).write_text(model.booster_.model_to_string(), encoding="utf-8")


def materialize(matrix, indices, path, preprocessing=None, batch_size=100_000):
    ncols = len(preprocessing.kept_indices_) if preprocessing else matrix.shape[1]
    target = np.lib.format.open_memmap(path, mode="w+", dtype="float64", shape=(len(indices), ncols))
    for start in range(0, len(indices), batch_size):
        block = matrix[indices[start:start + batch_size]]
        target[start:start + len(block)] = preprocessing.transform(block) if preprocessing else block
    target.flush()
    return target


def predict_batches(model, matrix, indices, preprocessing=None, batch_size=100_000):
    prediction = np.empty(len(indices), dtype="float64")
    for start in range(0, len(indices), batch_size):
        block = matrix[indices[start:start + batch_size]]
        if preprocessing:
            block = preprocessing.transform(block)
        if isinstance(model, lgb.LGBMRegressor):
            block = pd.DataFrame(block, columns=model.feature_name_)
        prediction[start:start + len(block)] = model.predict(block)
    return prediction


def run_fold(root, experiment_id, fold_id):
    folds_config, exp_config = configs(root)
    np.random.seed(exp_config["seed"])
    exp = next(e for e in exp_config["experiments"] if e["id"] == experiment_id)
    if experiment_id not in ["E000", "E001", "E002"] or exp["target"] != "raw":
        raise ValueError("Outside B2 scope")
    fold = next(f for f in folds_config["folds"] if f["id"] == fold_id)
    manifest = json.loads((root / "outputs/basic40_manifest.json").read_text(encoding="utf-8"))
    source = manifest["datasets"]["train"]["files"]
    matrix = np.load(source["matrix"]["path"], mmap_mode="r", allow_pickle=False)
    keys = np.load(source["keys"]["path"], mmap_mode="r", allow_pickle=False)
    labels = np.load(root / "outputs/baselines/cache/labels.npy", mmap_mode="r", allow_pickle=False)
    names = manifest["feature_names"]
    train_idx, valid_idx, split = split_fold(keys["trade_date"], labels, fold, folds_config["purge_rule"]["purge_trading_days"])
    out = root / "outputs/baselines" / experiment_id / fold_id
    out.mkdir(parents=True, exist_ok=True)
    if (out / "result.json").exists():
        raise FileExistsError("Completed fold exists; do not silently replace results")
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    print(f"START {experiment_id}/{fold_id}: train={len(train_idx):,}, valid={len(valid_idx):,}, purge={split['purge_dates']}", flush=True)
    started = time.perf_counter()
    preprocessor, model, training = None, None, None
    preprocess_metadata = None
    params = {}
    fit_seconds = 0.
    prep_seconds = 0.
    if experiment_id == "E000":
        predictions = np.asarray(matrix[valid_idx, names.index("ret_1")])
    else:
        if experiment_id == "E001":
            preprocessor = RidgePreprocessor(names).fit(matrix, train_idx)
            preprocess_metadata = preprocessor.metadata()
        train_file = out / "training_work.npy"
        training = materialize(matrix, train_idx, train_file, preprocessor)
        prep_seconds = time.perf_counter() - started
        model = make_model(experiment_id, exp_config)
        params = model.get_params(deep=False)
        fit_started = time.perf_counter()
        # No validation eval_set, early stopping, optimizer, manual solver or tuning.
        fit_input = pd.DataFrame(training, columns=names, copy=False) if experiment_id == "E002" else training
        model.fit(fit_input, np.asarray(labels[train_idx]))
        del fit_input
        fit_seconds = time.perf_counter() - fit_started
        print(f"FIT {experiment_id}/{fold_id} complete in {fit_seconds:.1f}s", flush=True)
        predictions = predict_batches(model, matrix, valid_idx, preprocessor)
        joblib.dump({"model": model, "preprocessor": preprocessor, "feature_names": names}, out / "model.joblib")
        if experiment_id == "E002":
            save_lightgbm_text(model, out / "model.txt")
        training._mmap.close()
        del training
        train_file.unlink()  # Only this known, generated, per-fold scratch file.
    generation_seconds = time.perf_counter() - started
    if experiment_id != "E000" and not np.isfinite(predictions).all():
        raise ValueError("Learned model produced nonfinite predictions")
    vkeys = keys[valid_idx]
    p = pd.DataFrame({"ts_code": vkeys["ts_code"], "trade_date": vkeys["trade_date"], "pred": predictions,
                      "y_ret_1d": np.asarray(labels[valid_idx]),
                      "flag_limit_up": matrix[valid_idx, names.index("flag_limit_up")]})
    evaluated, daily = evaluate(p)
    p.to_csv(out / "predictions.csv.gz", index=False, float_format="%.17g", compression={"method": "gzip", "mtime": 0})
    daily.to_csv(out / "daily_metrics.csv", index=False, float_format="%.17g")
    dropped = preprocessor.dropped_features_ if preprocessor else []
    run_id = f"B2_{experiment_id}_{fold_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    result = {
        "run_id": run_id, "experiment_id": experiment_id, "fold_id": fold_id,
        "timestamp": datetime.now(timezone.utc).isoformat(), "git_commit": git_commit,
        "data_version": manifest["data_version"], "feature_spec_version": manifest["feature_spec_version"],
        "protocol_version": exp_config["experiment_protocol_version"], "feature_set": exp["feature_set"],
        "n_features": 1 if experiment_id == "E000" else 40 - len(dropped), "target": exp["target"],
        "model": exp["model"], "configured_params": exp_config["model_catalog"].get(exp["model"], {}).get("params", {}),
        "resolved_params": params, "seed": exp_config["seed"], "split": split,
        "train_rows": split["train_rows"], "valid_rows": split["valid_rows"],
        "supervised_fit_rows": 0 if experiment_id == "E000" else split["train_rows"],
        "preprocessing": preprocess_metadata, "dropped_all_nan_features": dropped,
        "train_time_sec": fit_seconds, "preprocessing_materialization_sec": prep_seconds,
        "prediction_pipeline_sec": generation_seconds, "peak_memory_gb": peak_memory_gb(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                        "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "lightgbm": lgb.__version__},
        "config_sha256": {path.name: sha256(path) for path in (root / "config").glob("*.yaml")},
        "artifact_files": {path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
                           for path in out.iterdir() if path.is_file()},
        **evaluated,
    }
    save_json(out / "result.json", result)
    fields = exp_config["experiment_log"]["required_columns"]
    row = {key: result.get(key, "") for key in fields}
    row.update({key: split[key] for key in ["train_start", "train_end", "valid_start", "valid_end"]})
    row.update(evaluated["metrics"])
    row["params_json"] = json.dumps(params, sort_keys=True)
    row["dropped_all_nan_features_json"] = json.dumps(dropped)
    row["notes"] = f"PROVISIONAL {POLICY['id']}; purge={split['purge_dates']}; valid rows include missing labels; supervised_fit_rows={result['supervised_fit_rows']}; artifacts={out.relative_to(root).as_posix()}"
    logfile = root / exp_config["experiment_log"]["path"]
    exists = logfile.exists()
    with logfile.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"DONE {experiment_id}/{fold_id}: {json.dumps(clean_json(evaluated['metrics']))}", flush=True)
    return result


def summarize(root):
    results = []
    for e in ["E000", "E001", "E002"]:
        for f in ["F1", "F2", "F3"]:
            results.append(json.loads((root / "outputs/baselines" / e / f / "result.json").read_text(encoding="utf-8")))
    rows = []
    for e in ["E000", "E001", "E002"]:
        r = [x for x in results if x["experiment_id"] == e]
        scores = np.array([x["metrics"]["final_score"] for x in r], dtype=float)
        rows.append({"experiment_id": e, **{f"Score_{x['fold_id']}": x["metrics"]["final_score"] for x in r},
                     "MeanScore": float(scores.mean()), "WorstScore": float(scores.min()), "StdScore": float(scores.std(ddof=0))})
    save_json(root / "outputs/baseline_summary.json", {"evaluator_status": "provisional", "policy": POLICY, "summary": rows,
                                                       "fold_results": results})
    pd.DataFrame(rows).to_csv(root / "outputs/baseline_summary.csv", index=False, float_format="%.17g")
    lines = ["# B2 首轮 baseline（全部为 provisional）", "", "官方evaluate.py缺失，以下结果不能称为官方成绩；未修改冻结规范或调参。", "",
             "## 三折汇总", "", "| 实验 | F1 | F2 | F3 | 均值 | 最差折 | 标准差 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append("| " + row["experiment_id"] + " | " + " | ".join(f"{row[k]:.8f}" for k in ["Score_F1","Score_F2","Score_F3","MeanScore","WorstScore","StdScore"]) + " |")
    lines += ["", "## 每折指标", "", "| 实验/折 | RankIC | Top超额年化 | Turnover | Score | 非有限预测行 |", "|---|---:|---:|---:|---:|---:|"]
    for r in results:
        m = r["metrics"]
        lines.append(f"| {r['experiment_id']}/{r['fold_id']} | {m['rank_ic_mean']:.8f} | {m['annualized_top_excess_return']:.8f} | {m['mean_turnover']:.8f} | {m['final_score']:.8f} | {r['coverage']['nonfinite_predictions']} |")
    lines += ["", "## 时间切分与 purge", "", "| 折 | purge日期 | purge行数 | 有效训练行 | 验证全部行 | 验证有限标签行 |", "|---|---|---:|---:|---:|---:|"]
    for r in results[:3]:
        s = r["split"]
        lines.append(f"| {r['fold_id']} | {s['purge_dates']} | {s['purge_rows']} | {s['train_rows']} | {s['valid_rows']} | {s['valid_finite_label_rows']} |")
    lines += ["", "## 解释与限制", "",
              "- E000不拟合模型，原样pred=ret_1；其非有限预测不会填充，暂按provisional约定从各指标排除。与模型基线的评分覆盖范围可能不同。",
              "- Turnover不按标签缺失过滤；只排除涨停及非有限预测。每折从第二个验证日期开始计算。",
              "- Ridge使用官方scikit-learn；逐折全NaN列删除及中位数/缩放均仅fit于purge后的有限标签训练行。详情见各fold result.json。",
              "- LightGBM使用官方lightgbm和冻结参数，800轮；无验证eval_set、early stopping或调参。",
              "- daily_metrics.csv保存逐日指标及覆盖率；predictions.csv.gz保存全部验证键、原预测、标签和涨停标志。大预测/模型文件仅保存在本地，哈希随result.json提交。",
              "- 模型选择应结合三折均值、最差折和稳定性；本轮仅报告，不据此调整特征、目标、参数或执行后续阶段。",
              "- 完整评分边界约定见docs/PROVISIONAL_EVALUATOR.md；全部9项规范指标、样本数、资源用量和实现版本见baseline_summary.json及experiment_log.csv。", ""]
    (root / "outputs/baseline_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--experiment", choices=["E000", "E001", "E002"])
    parser.add_argument("--fold", choices=["F1", "F2", "F3"])
    args = parser.parse_args()
    root = args.root.resolve()
    if args.experiment and args.fold:
        run_fold(root, args.experiment, args.fold)
        return
    if args.experiment or args.fold:
        parser.error("experiment and fold must be supplied together")
    prepare_inputs(root)
    # Sequential isolated official-library fits bound memory and preserve order.
    for e in ["E000", "E001", "E002"]:
        for f in ["F1", "F2", "F3"]:
            if (root / "outputs/baselines" / e / f / "result.json").exists():
                raise FileExistsError("Existing completed baseline run; explicit review required before replacing it")
            subprocess.run([sys.executable, "-m", "stock_prediction.baselines", "--root", str(root), "--experiment", e, "--fold", f], check=True, cwd=root)
    summarize(root)


if __name__ == "__main__":
    main()
