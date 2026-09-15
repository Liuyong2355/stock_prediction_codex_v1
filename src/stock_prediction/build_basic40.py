"""Generate full Basic40 artifacts and descriptive audits, never model scores."""
import argparse
import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path

import numpy as np
import pandas as pd

from .audit import read_chunks
from .basic40 import KEYS, SOURCES, EPSILON, compute_basic40, load_contract, stock_batches


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_number(value):
    return float(value) if np.isfinite(value) else None


def review_sample(raw, features, position, reason):
    """Show small raw-input windows and independent scalar recomputations."""
    current = raw.iloc[position]
    history = raw.iloc[max(0, position - 5):position + 1]
    closes = history.close.to_numpy(dtype=float)
    last5 = raw.iloc[max(0, position - 4):position + 1]
    def complete(a, n):
        return len(a) == n and np.isfinite(a).all()
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        expected = {
            "ret_1": closes[-1] / closes[-2] - 1 if complete(closes[-2:], 2) else np.nan,
            "ret_5": closes[-1] / closes[0] - 1 if len(closes) == 6 and np.isfinite(closes[[0, -1]]).all() else np.nan,
            "madev_5": current.close / np.mean(last5.close) - 1 if complete(last5.close, 5) else np.nan,
            "volratio_5": current.vol / (np.mean(last5.vol) + EPSILON) if complete(last5.vol, 5) else np.nan,
            "stdret_5": np.std(closes[1:] / closes[:-1] - 1, ddof=0) if complete(closes, 6) else np.nan,
        }
    actual = features.iloc[position]
    checks = {name: {"expected": finite_number(value), "actual": finite_number(actual[name]),
                     "matches": bool(np.isclose(value, actual[name], atol=1e-10, rtol=1e-9, equal_nan=True))}
              for name, value in expected.items()}
    if not all(c["matches"] for c in checks.values()):
        raise AssertionError(f"Manual review recomputation failed: {checks}")
    return {"reason": reason, "ts_code": str(current.ts_code), "trade_date": int(current.trade_date),
            "raw_history": json.loads(history[KEYS + SOURCES].to_json(orient="records", double_precision=15)),
            "checks": checks}


def audit_matrix(matrix_path, keys_path, names):
    """Exact finite-only quantiles, one persisted float64 column at a time."""
    matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
    keys = np.load(keys_path, mmap_mode="r", allow_pickle=False)
    if matrix.shape != (len(keys), len(names)):
        raise AssertionError("Feature/key shape mismatch")
    rows = len(keys)
    statistics = {}
    for j, name in enumerate(names):
        column = matrix[:, j]
        finite = np.isfinite(column)
        values = np.asarray(column[finite])
        count = int(finite.sum())
        quantiles = np.quantile(values, [0, .01, .25, .5, .75, .99, 1]) if count else [np.nan] * 7
        extremes = []
        if count:
            for label, position in [("min", int(np.nanargmin(column))), ("max", int(np.nanargmax(column)))]:
                extremes.append({"kind": label, "row": position, "ts_code": str(keys[position]["ts_code"]),
                                 "trade_date": int(keys[position]["trade_date"]), "value": float(column[position])})
        statistics[name] = {"finite_count": count, "finite_ratio": count / rows,
                            "nan_count": int(np.isnan(column).sum()), "nan_ratio": int(np.isnan(column).sum()) / rows,
                            "inf_count": int(np.isinf(column).sum()),
                            "quantiles": {label: finite_number(v) for label, v in zip(["min", "p01", "p25", "p50", "p75", "p99", "max"], quantiles)},
                            "extreme_examples": extremes}
        if statistics[name]["inf_count"] or count + statistics[name]["nan_count"] != rows:
            raise AssertionError(f"Nonfinite output other than NaN: {name}")
    both = (matrix[:, names.index("flag_limit_up")] == 1) & (matrix[:, names.index("flag_limit_down")] == 1)
    return {"rows": rows, "n_features": matrix.shape[1], "dtype": str(matrix.dtype),
            "both_limit_flags_equal_one": int(both.sum()), "features": statistics}


def render_report(summary):
    lines = ["# Basic40 特征审计报告", "", f"生成时间：{summary['generated_at']}",
             f"数据版本：`{summary['data_version']}`", f"特征规范：{summary['feature_spec_version']}", "",
             "## 范围与结论", "",
             "- 完整train/test逐股票连续计算，输出40列float64；标识键单独保存，标签不进入特征。",
             "- lag按观测行计数；rolling含当行且要求完整有限窗口；std使用ddof=0。未补行、填充、截尾或标准化。",
             "- Basic40不含EMA、相关系数或横截面特征；这些Full147功能未实现。截面测试验证同日其他股票不会影响本股票Basic40。",
             "- 所有输出均为finite或NaN，没有inf；NaN保留，审计未按标签是否缺失删行。",
             "- 分位数是落盘后逐列全量finite值的精确分位数（NumPy默认linear插值），不是抽样估计。",
             "- 极端值仅描述、不判为错误、不裁剪；完整极值对应键见JSON。统计不用于任何预处理拟合。",
             "- 未运行E000–E007，未训练、调参或实现Full147。", "",
             "## warnings / unresolved observations", ""]
    lines += [f"- {w}" for w in summary["warnings"]]
    for split, data in summary["datasets"].items():
        lines += ["", f"## {split}（{data['rows']:,}行 × 40特征）", "",
                  "| 特征 | finite% | NaN% | min | p01 | p25 | median | p75 | p99 | max |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for name, stats in data["features"].items():
            q = stats["quantiles"]
            values = " | ".join("NaN" if q[k] is None else f"{q[k]:.8g}" for k in ["min", "p01", "p25", "p50", "p75", "p99", "max"])
            lines.append(f"| {name} | {stats['finite_ratio']:.6%} | {stats['nan_ratio']:.6%} | {values} |")
    lines += ["", "## 人工复核样例", "", "以下给出最多6行原始历史和独立标量计算；NaN用null表示。完整样例保存在JSON。"]
    for sample in summary["review_samples"]:
        lines += ["", f"### {sample['reason']}：{sample['ts_code']} / {sample['trade_date']}", "",
                  "```json", json.dumps(sample, ensure_ascii=False, indent=2), "```"]
    lines += ["", "## 落盘与可复现性", "",
              "输出路径、列序、矩阵形状及SHA-256见 outputs/basic40_manifest.json。矩阵和键文件位于outputs/features/basic40/，保留在本地、不提交Git。",
              "测试结果见outputs/basic40_test_results.txt。数据、规范和实现指纹见JSON；无新增依赖。", ""]
    return "\n".join(lines)


def build(root: Path, chunksize=200_000):
    started = time.perf_counter()
    contract_path = root / "config/features_v1.yaml"
    contract = load_contract(contract_path)
    names = contract["feature_sets"]["Basic40"]
    raw_audit = json.loads((root / "outputs/data_audit_summary.json").read_text(encoding="utf-8"))
    output = root / "outputs/features/basic40"
    output.mkdir(parents=True, exist_ok=True)
    paths, arrays, offsets = {}, {}, {"train": 0, "test": 0}
    for split, filename in [("train", "训练集.csv"), ("test", "测试集_X.csv")]:
        path = root / "data/raw" / filename
        if sha256(path) != raw_audit["datasets"][split]["sha256"]:
            raise ValueError("Raw data changed since accepted audit; audit it before feature generation")
        paths[split] = {"matrix": output / f"{split}_basic40.npy", "keys": output / f"{split}_keys.npy"}
        if any(p.exists() for p in paths[split].values()):
            raise FileExistsError("Completed Basic40 output exists; choose a fresh root or explicitly archive it before rebuilding")
        n = raw_audit["datasets"][split]["rows"]
        arrays[split] = {
            "matrix": np.lib.format.open_memmap(str(paths[split]["matrix"]) + ".partial", mode="w+", dtype="float64", shape=(n, 40), fortran_order=True),
            "keys": np.lib.format.open_memmap(str(paths[split]["keys"]) + ".partial", mode="w+", dtype=[("ts_code", "U16"), ("trade_date", "i8")], shape=(n,)),
        }
    train_iter = stock_batches(read_chunks(root / "data/raw/训练集.csv", True, chunksize))
    test_iter = stock_batches(read_chunks(root / "data/raw/测试集_X.csv", False, chunksize))
    samples = []
    selected_reasons = set()
    preserved_flags = {"train": True, "test": True}
    for i, pair in enumerate(zip_longest(train_iter, test_iter), 1):
        train_item, test_item = pair
        if train_item is None or test_item is None or train_item[0] != test_item[0]:
            raise ValueError("Train/test stock sets/order differ from accepted audit")
        code, train = train_item
        _, test = test_item
        if len(code) > 16 or train.trade_date.max() >= test.trade_date.min():
            raise ValueError("Invalid key length or train/test temporal overlap")
        raw = pd.concat([train, test], ignore_index=True)
        feature = compute_basic40(raw, contract)
        if not feature[KEYS].equals(raw[KEYS]):
            raise AssertionError("Feature key alignment changed")
        for split, begin, end in [("train", 0, len(train)), ("test", len(train), len(raw))]:
            values = feature.iloc[begin:end]
            pos = offsets[split]
            arrays[split]["matrix"][pos:pos + len(values)] = values[names].to_numpy()
            arrays[split]["keys"]["ts_code"][pos:pos + len(values)] = code
            arrays[split]["keys"]["trade_date"][pos:pos + len(values)] = values.trade_date
            offsets[split] += len(values)
            for flag in ["flag_limit_up", "flag_limit_down"]:
                preserved_flags[split] &= np.array_equal(values[flag], raw.iloc[begin:end][flag], equal_nan=True)
        candidates = {"训练初始窗口": 4, "60期收益首次具备历史": 60, "测试首日继承训练历史": len(train)} if i == 1 else {}
        zero = np.flatnonzero((raw.vol == 0) & (raw.amount == 0) & np.isfinite(raw.close))
        if len(zero):
            candidates["零量额保留"] = int(zero[0])
        both = np.flatnonzero((raw.flag_limit_up == 1) & (raw.flag_limit_down == 1))
        if len(both):
            candidates["涨跌停同时为1保留"] = int(both[0])
        after_nan = np.flatnonzero(np.isfinite(raw.close) & raw.close.shift(1).isna() & (raw.index > 0))
        if len(after_nan):
            candidates["缺失后首个有效价格仍不跳过空行"] = int(after_nan[0])
        for reason, position in candidates.items():
            if reason not in selected_reasons and position < len(raw):
                samples.append(review_sample(raw, feature, position, reason))
                selected_reasons.add(reason)
        if i % 500 == 0:
            print(f"Computed {i} stocks; rows {offsets}", flush=True)
    for split in arrays:
        if offsets[split] != raw_audit["datasets"][split]["rows"] or not preserved_flags[split]:
            raise AssertionError("Row count or flag preservation failed")
        for kind, array in arrays[split].items():
            array.flush()
            array._mmap.close()
            Path(str(paths[split][kind]) + ".partial").replace(paths[split][kind])
    print("Auditing persisted Basic40 columns", flush=True)
    datasets = {split: audit_matrix(p["matrix"], p["keys"], names) for split, p in paths.items()}
    for split in datasets:
        expected_both = raw_audit["datasets"][split]["flag_cross_field_consistency"]["both_limit_flags_equal_one"]["count"]
        if datasets[split]["both_limit_flags_equal_one"] != expected_both:
            raise AssertionError("Cross-field flags changed")
    for split, filename in [("train", "训练集.csv"), ("test", "测试集_X.csv")]:
        if sha256(root / "data/raw" / filename) != raw_audit["datasets"][split]["sha256"]:
            raise AssertionError("Raw bytes changed during build")
    try:
        base_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except subprocess.CalledProcessError:
        base_commit = None
    summary = {
        "schema_version": "1.0", "feature_spec_version": contract["feature_spec_version"],
        "generated_at": datetime.now(timezone.utc).isoformat(), "data_version": raw_audit["data_version"],
        "base_git_commit": base_commit, "runtime": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
        "feature_names": names, "config_sha256": sha256(contract_path),
        "feature_doc_sha256": sha256(root / "docs/FEATURE_SPEC_V1.md"),
        "implementation_sha256": {p.name: sha256(p) for p in [Path(__file__), Path(__file__).with_name("basic40.py")]},
        "datasets": datasets, "review_samples": samples,
        "validation": {"keys_match_raw": True, "all_raw_rows_preserved": True,
                       "flags_preserved": preserved_flags, "raw_sha256_unchanged": True,
                       "cross_section_features_in_basic40": 0, "ema_features_in_basic40": 0},
        "warnings": ["NaN来自原始缺失与历史/完整窗口不足，不填充或按标签删行。",
                     "涨跌停同时为1的记录原样保留，仅为未解释观测。",
                     "极端收益和量额比原样保留；没有裁剪阈值或异常值重编码。"],
        "elapsed_seconds": time.perf_counter() - started,
    }
    manifest = {"feature_set": "Basic40", "feature_spec_version": contract["feature_spec_version"],
                "data_version": raw_audit["data_version"], "feature_names": names,
                "storage": "float64 Fortran-order NumPy matrix plus aligned structured NumPy keys; allow_pickle=False",
                "datasets": {}}
    for split, data in paths.items():
        manifest["datasets"][split] = {"rows": offsets[split], "columns": 40, "files": {
            kind: {"path": str(path.resolve()), "sha256": sha256(path), "size_bytes": path.stat().st_size}
            for kind, path in data.items()}}
    for filename, content in [("basic40_audit_summary.json", summary), ("basic40_manifest.json", manifest)]:
        (root / "outputs" / filename).write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (root / "outputs/basic40_audit_report.md").write_text(render_report(summary), encoding="utf-8")
    print(f"Basic40 complete: {offsets}; audit and manifest saved", flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--chunksize", type=int, default=200_000)
    args = parser.parse_args()
    build(args.root.resolve(), args.chunksize)
